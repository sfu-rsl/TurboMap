#include "Kernels/CudaKeyframeDrawer.h"
#include <csignal> 

MAPPING_DATA_WRAPPER::CudaKeyframe *CudaKeyframeDrawer::d_keyframes, *CudaKeyframeDrawer::h_keyframes;
std::unordered_map<long unsigned int, MAPPING_DATA_WRAPPER::CudaKeyframe*> CudaKeyframeDrawer::id_to_kf;
int CudaKeyframeDrawer::num_keyframes = 0;
bool CudaKeyframeDrawer::memory_is_initialized = false;

void CudaKeyframeDrawer::initializeMemory(){    
    checkCudaError(cudaMallocHost((void**)&h_keyframes, CUDA_KEYFRAME_DRAWER_STORAGE * sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe)), "[CudaKeyframeDrawer::] Failed to allocate memory for h_keyframes");  
    for (int i = 0; i < CUDA_KEYFRAME_DRAWER_STORAGE; ++i) {
        h_keyframes[i] = MAPPING_DATA_WRAPPER::CudaKeyframe();
    }

    checkCudaError(cudaMalloc((void**)&d_keyframes, CUDA_KEYFRAME_DRAWER_STORAGE * sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe)), "[CudaKeyframeDrawer::] Failed to allocate memory for d_keyframes");
}

MAPPING_DATA_WRAPPER::CudaKeyframe* CudaKeyframeDrawer::addCudaKeyframe(ORB_SLAM3::KeyFrame* KF){
    if (!memory_is_initialized) {
        initializeMemory();
    }

    if (num_keyframes > CUDA_KEYFRAME_DRAWER_STORAGE) {
        cout << "[ERROR] CudaKeyframeDrawer::addCudaKeyframe: ] number of keyframes: " << num_keyframes << " is greater than CUDA_KEYFRAME_DRAWER_STORAGE: " << CUDA_KEYFRAME_DRAWER_STORAGE << "\n";
        raise(SIGSEGV);
    }

    h_keyframes[num_keyframes].setMemory(KF);
    checkCudaError(cudaMemcpy(&d_keyframes[num_keyframes], &h_keyframes[num_keyframes], sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe), cudaMemcpyHostToDevice), "[CudaKeyframeDrawer::] Failed to copy individual element to d_keyframes");
    id_to_kf.emplace(KF->mnId, &d_keyframes[num_keyframes]);
    num_keyframes += 1;

    return &d_keyframes[num_keyframes-1];
}

MAPPING_DATA_WRAPPER::CudaKeyframe* CudaKeyframeDrawer::getCudaKeyframe(long unsigned int mnId){
    auto it = id_to_kf.find(mnId);
    if (it != id_to_kf.end()) {
        return it->second;
    }
    return nullptr;
}
