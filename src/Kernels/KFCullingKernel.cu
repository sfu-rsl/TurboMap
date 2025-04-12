#include "Kernels/KFCullingKernel.h"
#include "Kernels/MappingKernelController.h"
#include <csignal> 

void KFCullingKernel::initialize(){
    if (memory_is_initialized) {
        return;
    }

    int maxFeatures = CudaUtils::nFeatures_with_th;
    size_t maxNeighborCount = 300;
    size_t mapPointVecSize;
    if (CudaUtils::cameraIsFisheye)
        mapPointVecSize = maxFeatures*2;
    else
        mapPointVecSize = maxFeatures;

    checkCudaError(cudaMalloc((void**)&d_keyframes, MAX_NUM_KEYFRAMES * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*)), "[KFCullingKernel::] Failed to allocate memory for d_keyframes");   
    checkCudaError(cudaMalloc((void**)&d_nRedundantObservations, MAX_NUM_KEYFRAMES * sizeof(int)), "[KFCullingKernel::] Failed to allocate memory for d_nRedundantObservations"); 
    checkCudaError(cudaMalloc((void**)&d_nMPs, MAX_NUM_KEYFRAMES * sizeof(int)), "[KFCullingKernel::] Failed to allocate memory for d_nMPs"); 
    checkCudaError(cudaMalloc((void**)&d_neighFramesMapPointsCorrect, maxNeighborCount * mapPointVecSize * sizeof(bool)), "Failed to allocate device vector d_neighFramesMapPointsCorrect");
    checkCudaError(cudaMalloc((void**)&d_neighKFsMapPoints, maxNeighborCount * mapPointVecSize * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint)), "Failed to allocate memory for d_neighKFsMapPoints");

    memory_is_initialized = true;
}

__global__ void keyframeCullingKernel() {
    
}

void KFCullingKernel::launch(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int* h_nMPs, int* h_nRedundantObservations) {

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startPreProcess = std::chrono::steady_clock::now();
#endif

    if(!memory_is_initialized){
        initialize();
    }

    int vpLocalKeyFrames_size = vpLocalKeyFrames.size();
    if (vpLocalKeyFrames_size == 0)
        return;

    for(int i=0; i < vpLocalKeyFrames_size; i++) {

        ORB_SLAM3::KeyFrame* pKF = vpLocalKeyFrames[i];

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad()) {
            MAPPING_DATA_WRAPPER::CudaKeyFrame* d_kf = nullptr;
            checkCudaError(cudaMemcpy(&d_keyframes[i], &d_kf, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*), cudaMemcpyHostToDevice), "[KFCullingKernel::] Failed to set d_keyframes[i] to null");
        }
        
        else {

            MAPPING_DATA_WRAPPER::CudaKeyFrame* d_kf = CudaKeyFrameStorage::getCudaKeyFrame(pKF->mnId);
            if (d_kf == nullptr) {
                cout << "[ERROR] KFCullingKernel::launch: ] CudaKeyFrameStorage doesn't have the keyframe: " << pKF->mnId << "\n";
                MappingKernelController::shutdownKernels(true, true);
                exit(EXIT_FAILURE);
            }

            checkCudaError(cudaMemcpy(&d_keyframes[i], &d_kf, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*), cudaMemcpyHostToDevice), "[KFCullingKernel::] Failed to copy d_kf from drawer to d_keyframes");
        }
    }

    int maxFeatures = CudaUtils::nFeatures_with_th;
    size_t mapPointVecSize;
    if (CudaUtils::cameraIsFisheye)
        mapPointVecSize = maxFeatures*2;
    else
        mapPointVecSize = maxFeatures;

    bool neighFramesMapPointsCorrect[vpLocalKeyFrames_size][mapPointVecSize];
    MAPPING_DATA_WRAPPER::CudaMapPoint wrappedNeighKFMapPoints[vpLocalKeyFrames_size][mapPointVecSize];
    for (int i = 0; i < vpLocalKeyFrames_size; i++) {
        vpLocalKeyFrames[i]->GetMapPointCorrectness(neighFramesMapPointsCorrect[i]);
        std::vector<ORB_SLAM3::MapPoint*> mapPoints = vpLocalKeyFrames[i]->GetMapPointMatches();
        for (int j = 0; j < mapPoints.size(); j++)
            wrappedNeighKFMapPoints[i][j] = MAPPING_DATA_WRAPPER::CudaMapPoint(mapPoints[j]);
    }
    
    checkCudaError(cudaMemcpy(d_neighKFsMapPoints, wrappedNeighKFMapPoints, sizeof(wrappedNeighKFMapPoints), cudaMemcpyHostToDevice), "Failed to copy vector wrappedNeighKFMapPoints from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesMapPointsCorrect, neighFramesMapPointsCorrect, sizeof(neighFramesMapPointsCorrect), cudaMemcpyHostToDevice), "Failed to copy vector neighFramesMapPointsCorrect from host to device");
    
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endPreProcess = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startKernel = std::chrono::steady_clock::now();
#endif

    int thObs = 3;
    int blockSize = 1024;
    int numBlocks = vpLocalKeyFrames_size;
    keyframeCullingKernel<<<numBlocks, blockSize>>>();
    checkCudaError(cudaDeviceSynchronize(), "[KFCullingKernel:] Kernel launch failed");  

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endKernel = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpyToCPU = std::chrono::steady_clock::now();
#endif

    checkCudaError(cudaMemcpy(h_nMPs, d_nMPs, vpLocalKeyFrames_size * sizeof(int), cudaMemcpyDeviceToHost), "[KFCullingKernel::] Failed to copy d_nMPs to h_nMPs");
    checkCudaError(cudaMemcpy(h_nRedundantObservations, d_nRedundantObservations, vpLocalKeyFrames_size * sizeof(int), cudaMemcpyDeviceToHost), "[KFCullingKernel::] Failed to copy d_nRedundantObservations to h_nRedundantObservations");

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point endMemcpyToCPU = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();

    double preProcess = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endPreProcess - startPreProcess).count();
    double kernel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endKernel - startKernel).count();
    double memcpyToCPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpyToCPU - startMemcpyToCPU).count();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    pre_process_time.emplace_back(frameCounter, preProcess);
    kernel_exec_time.emplace_back(frameCounter, kernel);
    output_data_transfer_time.emplace_back(frameCounter, memcpyToCPU);
    total_exec_time.emplace_back(frameCounter, total);

    frameCounter++;
#endif
} 

void KFCullingKernel::KFCullingOrig(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int thObs, bool mbMonocular) {

    for(int kf_idx = 0; kf_idx < vpLocalKeyFrames.size(); kf_idx++)
    {
        ORB_SLAM3::KeyFrame* pKF = vpLocalKeyFrames[kf_idx];

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad()) {
            //printf("kf_idx: %d, continue1\n", kf_idx);
            continue;
        }
        const vector<ORB_SLAM3::MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;

        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
            if (!pMP) {
                //printf("kf_idx: %d, mp_idx: %d, continue 2\n", kf_idx, i);
                continue;
            }

            if(pMP->isBad()) {
                //printf("kf_idx: %d, mp_idx: %d, continue 3\n", kf_idx, i);
                continue;
            }

            if(!mbMonocular)
            {
                if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0) {
                    //printf("kf_idx: %d, mp_idx: %d, continue 4\n", kf_idx, i);
                    continue;
                }
            }

            // printf("kf_idx: %d, adding mp_idx: %d\n", kf_idx, i);

            nMPs++;

            if(pMP->Observations() <= thObs) {
                //printf("kf_idx: %d, mp_idx: %d, continue 5\n", kf_idx, i);
                continue;
            }

            const int &scaleLevel = (pKF -> NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                            : (i < pKF -> NLeft) ? pKF -> mvKeys[i].octave
                                                                                : pKF -> mvKeysRight[i].octave;

            // printf("kf_idx: %d, mp_idx: %d, scaleLevel: %d\n", kf_idx, i, scaleLevel);

            const map<ORB_SLAM3::KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
            int nObs=0;
            for(map<ORB_SLAM3::KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                ORB_SLAM3::KeyFrame* pKFi = mit->first;
                if(pKFi==pKF) {
                    // printf("    kf_idx: %d, mp_idx: %d, pKFi: %d, continue 6\n", kf_idx, i, pKFi->mnId);
                    continue;
                }
                tuple<int,int> indexes = mit->second;
                int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                int scaleLeveli = -1;
                if(pKFi -> NLeft == -1)
                    scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
                else {
                    if (leftIndex != -1) {
                        scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                    }
                    if (rightIndex != -1) {
                        int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
                        scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                        : scaleLeveli;
                    }
                }

                // printf("    kf_idx: %d, mp_idx: %d, pKFi: %d, scaleLevel: %d, scaleLeveli: %d\n", kf_idx, i, pKFi->mnId, scaleLevel, scaleLeveli);

                if(scaleLeveli<=scaleLevel+1)
                {
                    nObs++;
                    // printf("kf_idx: %d, mp_idx: %d, pKFi: %d, nObs: %d\n", kf_idx, i, pKFi->mnId, nObs);
                    if(nObs>thObs)
                        break;
                }
            }
            if(nObs>thObs)
            {
                // printf("kf_idx: %d, adding mp_idx as redundant obs: %d\n", kf_idx, i);
                nRedundantObservations++;
            }

        }

        printf("KF: %d, nMPs: %d, nRedun: %d\n", kf_idx, nMPs, nRedundantObservations);
    }
}

void KFCullingKernel::shutdown(){
    if (!memory_is_initialized) 
        return;
    
    cudaFree(d_keyframes);
    cudaFree(d_nRedundantObservations);
    cudaFree(d_nMPs);
    cudaFree(d_neighFramesMapPointsCorrect);
    cudaFree(d_neighKFsMapPoints);
}

void KFCullingKernel::saveStats(const std::string &file_path) {
    std::string data_path = file_path + "/KFCullingKernel/";
    std::cout << "[KFCullingKernel:] writing stats data into file: " << data_path << '\n';
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[KFCullingKernel:] Error creating directory: " << strerror(errno) << std::endl;
    }
    std::ofstream myfile;

    myfile.open(data_path + "/pre_process_time.txt");
    for (const auto& p : pre_process_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
    
    myfile.open(data_path + "/kernel_exec_time.txt");
    for (const auto& p : kernel_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
    
    myfile.open(data_path + "/output_data_transfer_time.txt");
    for (const auto& p : output_data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/total_exec_time.txt");
    for (const auto& p : total_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
}