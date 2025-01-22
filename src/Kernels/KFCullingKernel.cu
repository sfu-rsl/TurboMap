#include "Kernels/KFCullingKernel.h"

void KFCullingKernel::initialize(){
    if (memory_is_initialized) {
        return;
    }

    checkCudaError(cudaMalloc((void**)&d_keyframes, MAX_NUM_KEYFRAMES * sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe)), "[KFCullingKernel::] Failed to allocate memory for d_keyframes");   

    checkCudaError(cudaMallocHost((void**)&h_keyframes, MAX_NUM_KEYFRAMES * sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe)), "[KFCullingKernel::] Failed to allocate memory for h_keyframes");   
    for (int i = 0; i < MAX_NUM_KEYFRAMES; ++i) {
        h_keyframes[i] = MAPPING_DATA_WRAPPER::CudaKeyframe();
    }

    memory_is_initialized = true;
}

void KFCullingKernel::launch(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames){
    int KF_count = 0;
    for(vector<ORB_SLAM3::KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        ORB_SLAM3::KeyFrame* pKF = *vit;
        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        h_keyframes[KF_count].setMemory(pKF);
        KF_count++;
    }
    checkCudaError(cudaMemcpy(&d_keyframes, &h_keyframes, vpLocalKeyFrames.size() * sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe), cudaMemcpyHostToDevice), "[KFCullingKernel::] Failed to copy h_keyframes to d_keyframes");
} 

void KFCullingKernel::shutdown(){
    if (!memory_is_initialized) 
        return;

    for (int i = 0; i < MAX_NUM_KEYFRAMES; ++i) {
        h_keyframes[i].freeMemory();
    }
    cudaFree(d_keyframes);
    cudaFreeHost(h_keyframes);
}