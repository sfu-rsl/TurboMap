#include "Kernels/KFCullingKernel.h"
#include "Kernels/CudaKeyframeDrawer.h"
#include <csignal> 

void KFCullingKernel::initialize(){
    if (memory_is_initialized) {
        return;
    }

    checkCudaError(cudaMalloc((void**)&d_keyframes, MAX_NUM_KEYFRAMES * sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe*)), "[KFCullingKernel::] Failed to allocate memory for d_keyframes");   

    // checkCudaError(cudaMallocHost((void**)&h_keyframes, MAX_NUM_KEYFRAMES * sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe)), "[KFCullingKernel::] Failed to allocate memory for h_keyframes");   
    // for (int i = 0; i < MAX_NUM_KEYFRAMES; ++i) {
    //     h_keyframes[i] = MAPPING_DATA_WRAPPER::CudaKeyframe();
    // }

    checkCudaError(cudaMalloc((void**)&d_nRedundantObservations, MAX_NUM_KEYFRAMES * sizeof(int)), "[KFCullingKernel::] Failed to allocate memory for d_nRedundantObservations"); 
    checkCudaError(cudaMalloc((void**)&d_nMPs, MAX_NUM_KEYFRAMES * sizeof(int)), "[KFCullingKernel::] Failed to allocate memory for d_nMPs"); 

    memory_is_initialized = true;
}

__global__ void keyframeCullingKernel(MAPPING_DATA_WRAPPER::CudaKeyframe** d_keyframes, int numKeyframes, int thObs, bool mbMonocular,
                                    int* d_nMPs, int* d_nRedundantObservations) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numKeyframes) {

        MAPPING_DATA_WRAPPER::CudaKeyframe* pKF = d_keyframes[idx];

        if (pKF == nullptr) return;

        d_nRedundantObservations[idx] = 0;
        d_nMPs[idx] = 0;

        MAPPING_DATA_WRAPPER::CudaMapPoint** vpMapPoints = pKF->mvpMapPoints;

        for(int i = 0; i < pKF->mvpMapPoints_size; i++) {
            
            MAPPING_DATA_WRAPPER::CudaMapPoint* pMP = vpMapPoints[i];

            if (pMP == nullptr) continue;

            if (pMP->isEmpty) continue;
            
            if (pMP->mbBad) continue;

            if(!mbMonocular)
            {
                if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                    continue;
            }

            // printf("[==GPU RUN==] [nMPs++] KF: %lu, MP: %lu\n", pKF->mnId, pMP->mnId);

            d_nMPs[idx] += 1;

            printf("3. pMP->mnId: %lu pMP->nObs: %d thObs: %d\n", pMP->mnId, pMP->nObs, thObs);

            if(pMP->nObs <= thObs) continue;

            
            int scaleLevel = (pKF->NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                            : (i < pKF->NLeft) ? pKF->mvKeys[i].octave
                                                                                : pKF->mvKeysRight[i].octave;
                                                                        
            // printf("pMP->mObservations_size: %d\n", pMP->mObservations_size);
            int nObs=0;
            for(int j = 0; j < pMP->mObservations_size; j++) {

                MAPPING_DATA_WRAPPER::CudaKeyframe* pKFi = pMP->mObservations_dkf[j];

                // printf("PKFi: %p\n", pKFi);
                // printf("PKFi->mnId: %d\n", pKFi->mnId);

                if(pKFi->mnId==pKF->mnId)
                    continue;

                int leftIndex = pMP->mObservations_leftIdx[j];
                int rightIndex =  pMP->mObservations_rightIdx[j];
                int scaleLeveli = -1;

                if(pKFi->NLeft == -1)
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
                
                // if(pKF->mnId == 4 && pMP->mnId == 12) {
                //     printf("[==GPU RUN==] KF: %lu, MP: %lu, j: %d, pKFi: %lu, leftIndex: %d, rightIndex: %d\n", pKF->mnId, pMP->mnId, j, (void*)pKFi->mnId, leftIndex, rightIndex);
                //     printf("[==GPU RUN==] KF: %lu, MP: %lu, j: %d, scaleLeveli: %d, scaleLevel: %d\n", pKF->mnId, pMP->mnId, j, scaleLeveli, scaleLevel);
                // }
                // if(pKF->mnId == 4 && pMP->mnId == 13) {
                //     printf("[==GPU RUN==] KF: %lu, MP: %lu, j: %d, pKFi: %lu, leftIndex: %d, rightIndex: %d\n", pKF->mnId, pMP->mnId, j, (void*)pKFi->mnId, leftIndex, rightIndex);
                //     printf("[==GPU RUN==] KF: %lu, MP: %lu, j: %d, scaleLeveli: %d, scaleLevel: %d\n", pKF->mnId, pMP->mnId, j, scaleLeveli, scaleLevel);
                // }

                if(scaleLeveli<=scaleLevel+1) {
                    nObs++;
                    if(nObs>thObs)
                        break;
                }
            }

            if(nObs>thObs)
            {
                // printf("[==GPU RUN==] [nRedundantObservations++] KF: %lu, MP: %lu\n", pKF->mnId, pMP->mnId);
                d_nRedundantObservations[idx] += 1;
            }
        }
    }
}

void validateKFInput_CPU(ORB_SLAM3::KeyFrame* KF) {
    vector<ORB_SLAM3::MapPoint*> h_mapPoints = KF->GetMapPointMatches();
    int mvpMapPoints_size = h_mapPoints.size();
    for (int i = 0; i < mvpMapPoints_size; ++i) {
        if(h_mapPoints[i]) {
            std::map<ORB_SLAM3::KeyFrame*,std::tuple<int,int>> observations = h_mapPoints[i]->GetObservations();
            for(map<ORB_SLAM3::KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                ORB_SLAM3::KeyFrame* pKFi = mit->first;
                printf("[CPU::] pKFi mnId: %lu\n", pKFi->mnId);
            }           
        }
        else {
            printf("[CPU::] i: %d, mp: is empty\n", i);
        }
    }
}

__global__ void validateKFInput_GPU(MAPPING_DATA_WRAPPER::CudaKeyframe** d_keyframes, int idx) {
    printf("***********************KF CULLING KERNEL******************************\n");
    MAPPING_DATA_WRAPPER::CudaKeyframe* KF = d_keyframes[idx];
    for (int i = 0; i < KF->mvpMapPoints_size; ++i) {
        //  printf("1\n");
        MAPPING_DATA_WRAPPER::CudaMapPoint* mp = KF->mvpMapPoints[i];
        //  printf("2\n");
        if (mp->isEmpty) {
            printf("[GPU::] i: %d, mp: is empty\n", i);
        } else {
            // printf("[GPU::] i: %d, mp: %lu\n", i, mp->mnId);
            MAPPING_DATA_WRAPPER::CudaKeyframe** mObservations_dkf = mp->mObservations_dkf;
            for (int j = 0; j < mp->mObservations_size; ++j) {
                int leftIdx = mp->mObservations_leftIdx[j];
                // printf("[GPU::] mp: %lu, leftIdx: %d\n", mp->mnId, leftIdx);
                MAPPING_DATA_WRAPPER::CudaKeyframe* pKFi = mp->mObservations_dkf[j];
                printf("    [GPU::] j:%d. mp: %lu, pKFi ptr: %p\n", j, mp->mnId, (void*)pKFi);
                printf("    [GPU::] j:%d, mp: %lu, pKFi mnId: %lu\n", j, mp->mnId, (void*)pKFi->mnId);
            }
        }
    }
}

void KFCullingKernel::launch(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int* h_kf_count, long unsigned int* h_indices,
                                                                            int* h_nMPs, int* h_nRedundantObservations){
    if(!memory_is_initialized){
        initialize();
    }

    int KF_count = 0;
    int vpLocalKeyFrames_size = vpLocalKeyFrames.size();
    for(int i=0; i < vpLocalKeyFrames_size; i++) {

        ORB_SLAM3::KeyFrame* pKF = vpLocalKeyFrames[i];

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad()) {
            MAPPING_DATA_WRAPPER::CudaKeyframe* d_kf = nullptr;
            checkCudaError(cudaMemcpy(&d_keyframes[KF_count], &d_kf, sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe*), cudaMemcpyHostToDevice), "[KFCullingKernel::] Failed to set d_keyframes[i] to null");
            KF_count++;
            continue;
        }

        MAPPING_DATA_WRAPPER::CudaKeyframe* d_kf = CudaKeyframeDrawer::getCudaKeyframe(pKF->mnId);
        if (d_kf == nullptr) {
            cout << "[ERROR] KFCullingKernel::launch: ] CudaKeyframeDrawer doesn't have the keyframe: " << pKF->mnId << "\n";
            raise(SIGSEGV);
        }

        checkCudaError(cudaMemcpy(&d_keyframes[KF_count], &d_kf, sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe*), cudaMemcpyHostToDevice), "[KFCullingKernel::] Failed to copy d_kf from drawer to d_keyframes");
        KF_count++;
    }

    int nObs = 3;
    const int thObs=nObs;
    int blockSize = 1;
    int numBlocks = 1;
    // int numBlocks = (numPoints + blockSize - 1) / blockSize;
    keyframeCullingKernel<<<numBlocks, blockSize>>>(d_keyframes, vpLocalKeyFrames_size, thObs, CudaUtils::isMonocular,
                                    d_nMPs, d_nRedundantObservations);
    checkCudaError(cudaDeviceSynchronize(), "[KFCullingKernel:] Kernel launch failed");  

    checkCudaError(cudaMemcpy(h_nMPs, d_nMPs, vpLocalKeyFrames_size * sizeof(int), cudaMemcpyDeviceToHost), "[KFCullingKernel::] Failed to copy d_nMPs to h_nMPs");
    checkCudaError(cudaMemcpy(h_nRedundantObservations, d_nRedundantObservations, vpLocalKeyFrames_size * sizeof(int), cudaMemcpyDeviceToHost), "[KFCullingKernel::] Failed to copy d_nRedundantObservations to h_nRedundantObservations");
} 

void KFCullingKernel::shutdown(){
    if (!memory_is_initialized) 
        return;

    // for (int i = 0; i < MAX_NUM_KEYFRAMES; ++i) {
    //     h_keyframes[i].freeMemory();
    // }
    // cudaFreeHost(h_keyframes);
    cudaFree(d_keyframes);
    cudaFree(d_nRedundantObservations);
    cudaFree(d_nMPs);
}