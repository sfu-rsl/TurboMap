#include "Kernels/KFCullingKernel.h"
#include <csignal> 

void KFCullingKernel::initialize(){
    if (memory_is_initialized) {
        return;
    }

    checkCudaError(cudaMalloc((void**)&d_keyframes, MAX_NUM_KEYFRAMES * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*)), "[KFCullingKernel::] Failed to allocate memory for d_keyframes");   

    // checkCudaError(cudaMallocHost((void**)&h_keyframes, MAX_NUM_KEYFRAMES * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "[KFCullingKernel::] Failed to allocate memory for h_keyframes");   
    // for (int i = 0; i < MAX_NUM_KEYFRAMES; ++i) {
    //     h_keyframes[i] = MAPPING_DATA_WRAPPER::CudaKeyFrame();
    // }

    checkCudaError(cudaMalloc((void**)&d_nRedundantObservations, MAX_NUM_KEYFRAMES * sizeof(int)), "[KFCullingKernel::] Failed to allocate memory for d_nRedundantObservations"); 
    checkCudaError(cudaMalloc((void**)&d_nMPs, MAX_NUM_KEYFRAMES * sizeof(int)), "[KFCullingKernel::] Failed to allocate memory for d_nMPs"); 

    memory_is_initialized = true;
}

__global__ void keyframeCullingKernel(MAPPING_DATA_WRAPPER::CudaKeyFrame** d_keyframes, int numKeyframes, int thObs, bool mbMonocular,
                                    int* d_nMPs, int* d_nRedundantObservations) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numKeyframes) {

        MAPPING_DATA_WRAPPER::CudaKeyFrame* pKF = d_keyframes[idx];

        // printf("[==GPU RUN==] 0\n");

        if (pKF == nullptr) return;

        if (pKF->isEmpty) return;

        d_nMPs[idx] = 0;
        d_nRedundantObservations[idx] = 0;

        MAPPING_DATA_WRAPPER::CudaMapPoint** vpMapPoints = pKF->mvpMapPoints;

        for(int i = 0; i < pKF->mvpMapPoints_size; i++) {
            
            MAPPING_DATA_WRAPPER::CudaMapPoint* pMP = vpMapPoints[i];

            // printf("[==GPU RUN==] 1 KF: %lu\n", pKF->mnId);
            
            if (pMP == nullptr) continue;
            
            if (pMP->isEmpty) continue;

            if (pKF->mnId == 1) {
                printf(" ,[%d: %lu]", i, pMP->mnId);
            }
            
            if (pMP->mbBad) continue;

            if(!mbMonocular)
            {
                if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                    continue;
            }

            // printf("[==GPU RUN==] 2 KF: %lu\n", pKF->mnId);

            // printf("[==GPU RUN==] [nMPs++] KF: %lu, MP: %lu\n", pKF->mnId, pMP->mnId);

            d_nMPs[idx] += 1;


            // printf("3. pMP->mnId: %lu pMP->nObs: %d thObs: %d\n", pMP->mnId, pMP->nObs, thObs);

            if(pMP->nObs <= thObs) continue;

            
            int scaleLevel = (pKF->NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                            : (i < pKF->NLeft) ? pKF->mvKeys[i].octave
                                                                                : pKF->mvKeysRight[i].octave;
                                                                        
            // printf("pMP->mObservations_size: %d\n", pMP->mObservations_size);
            int nObs=0;
            for(int j = 0; j < pMP->mObservations_size; j++) {

                MAPPING_DATA_WRAPPER::CudaKeyFrame* pKFi = pMP->mObservations_dkf[j];

                if (pKFi == nullptr) continue;

                if (pKFi->isEmpty) continue;

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
        printf("\n=============GPU==============\n");
    }
}

void KFCullingKernel::launch(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int* h_nMPs, int* h_nRedundantObservations) {
    if(!memory_is_initialized){
        initialize();
    }

    int KF_count = 0;
    int vpLocalKeyFrames_size = vpLocalKeyFrames.size();
    for(int i=0; i < vpLocalKeyFrames_size; i++) {

        ORB_SLAM3::KeyFrame* pKF = vpLocalKeyFrames[i];

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad()) {
            MAPPING_DATA_WRAPPER::CudaKeyFrame* d_kf = nullptr;
            checkCudaError(cudaMemcpy(&d_keyframes[KF_count], &d_kf, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*), cudaMemcpyHostToDevice), "[KFCullingKernel::] Failed to set d_keyframes[i] to null");
            KF_count++;
            continue;
        }

        // cout << "\nOriginal: pKF " << endl;
        // printKeyframeCPU(pKF);
        // cout << endl;

        MAPPING_DATA_WRAPPER::CudaKeyFrame* d_kf = CudaKeyFrameDrawer::getCudaKeyFrame(pKF->mnId);
        cudaDeviceSynchronize();
        if (d_kf == nullptr) {
            cout << "[ERROR] KFCullingKernel::launch: ] CudaKeyFrameDrawer doesn't have the keyframe: " << pKF->mnId << "\n";
            raise(SIGSEGV);
        }

        // cout << "\nBefore copy: d_kf " << endl;
        // printKFSingleGPU<<<1,1>>>(d_kf);
        // cout << endl;
        
        checkCudaError(cudaMemcpy(&d_keyframes[KF_count], &d_kf, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*), cudaMemcpyHostToDevice), "[KFCullingKernel::] Failed to copy d_kf from drawer to d_keyframes");
        
        // cout << "\nAfter copy: d_keyframes[" << KF_count << "]" << endl;
        // printKFListGPU<<<1,1>>>(d_keyframes, KF_count);
        // cout << endl;

        KF_count++;
    }

    int nObs = 3;
    const int thObs=nObs;
    // int blockSize = 1;
    // int numBlocks = 1;
    int blockSize = 128;
    int numBlocks = (KF_count + blockSize - 1) / blockSize;
    keyframeCullingKernel<<<numBlocks, blockSize>>>(d_keyframes, vpLocalKeyFrames_size, thObs, CudaUtils::isMonocular, d_nMPs, d_nRedundantObservations);
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