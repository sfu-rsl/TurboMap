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

            if (pMP == nullptr) continue;
            
            if (pMP->isEmpty) continue;
            
            if (pMP->mbBad) continue;

            if(!mbMonocular)
            {
                if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                    continue;
            }

            d_nMPs[idx] += 1;

            if(pMP->nObs <= thObs) continue;

            
            int scaleLevel = (pKF->Nleft == -1) ? pKF->mvKeysUn[i].octave
                                                            : (i < pKF->Nleft) ? pKF->mvKeys[i].octave
                                                                                : pKF->mvKeysRight[i].octave;

            int nObs=0;
            for(int j = 0; j < pMP->mObservations_size; j++) {

                MAPPING_DATA_WRAPPER::CudaKeyFrame* pKFi = pMP->mObservations_dkf[j];

                if (pKFi == nullptr) continue;

                if (pKFi->isEmpty) continue;

                if(pKFi->mnId==pKF->mnId)
                    continue;

                int leftIndex = pMP->mObservations_leftIdx[j];
                int rightIndex =  pMP->mObservations_rightIdx[j];
                int scaleLeveli = -1;

                if(pKFi->Nleft == -1)
                    scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
                else {
                    if (leftIndex != -1) {
                        scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                    }
                    if (rightIndex != -1) {
                        int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->Nleft].octave;
                        scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                        : scaleLeveli;
                    }
                }

                if(scaleLeveli<=scaleLevel+1) {
                    nObs++;
                    if(nObs>thObs)
                        break;
                }
            }

            if(nObs>thObs)
            {
                d_nRedundantObservations[idx] += 1;
            }
        }
    }
}

void KFCullingKernel::launch(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int* h_nMPs, int* h_nRedundantObservations) {
    if(!memory_is_initialized){
        initialize();
    }

    int KF_count = 0;
    int vpLocalKeyFrames_size = vpLocalKeyFrames.size();

    if (vpLocalKeyFrames_size == 0)
        return;

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
    checkCudaError(cudaGetLastError(), "[KFCullingKernel:] Failed to launch kernel");
    checkCudaError(cudaDeviceSynchronize(), "[KFCullingKernel:] cudaDeviceSynchronize after kernel launch failed");  

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