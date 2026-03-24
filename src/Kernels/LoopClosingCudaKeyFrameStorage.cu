#include "Kernels/LoopClosingCudaKeyFrameStorage.h"
#include "Kernels/LoopClosingKernelController.h"

LOOP_DATA_WRAPPER::CudaKeyFrame *LoopClosingCudaKeyFrameStorage::d_keyframes, *LoopClosingCudaKeyFrameStorage::h_keyframes;
int LoopClosingCudaKeyFrameStorage::num_keyframes = 0;
bool LoopClosingCudaKeyFrameStorage::memory_is_initialized = false;
ckd_buffer_index_t LoopClosingCudaKeyFrameStorage::first_free_idx = 0;
std::unordered_map<long unsigned int, ckd_buffer_index_t> LoopClosingCudaKeyFrameStorage::mnId_to_idx;
std::queue<ckd_buffer_index_t> LoopClosingCudaKeyFrameStorage::free_idx;
std::mutex LoopClosingCudaKeyFrameStorage::mtx;


void LoopClosingCudaKeyFrameStorage::initializeMemory(){  
    if (memory_is_initialized) return;

    checkCudaError(cudaMallocHost((void**)&h_keyframes, CUDA_KEYFRAME_STORAGE_SIZE * sizeof(LOOP_DATA_WRAPPER::CudaKeyFrame)), "[CudaKeyFrameStorage::] Failed to allocate memory for h_keyframes");  
    for (int i = 0; i < CUDA_KEYFRAME_STORAGE_SIZE; ++i) {
        h_keyframes[i] = LOOP_DATA_WRAPPER::CudaKeyFrame();
    }

    checkCudaError(cudaMalloc((void**)&d_keyframes, CUDA_KEYFRAME_STORAGE_SIZE * sizeof(LOOP_DATA_WRAPPER::CudaKeyFrame)), "[CudaKeyFrameStorage::] Failed to allocate memory for d_keyframes");
    memory_is_initialized = true;
}


LOOP_DATA_WRAPPER::CudaKeyFrame* LoopClosingCudaKeyFrameStorage::addCudaKeyFrame(ORB_SLAM3::KeyFrame* KF) {
  
    if (!memory_is_initialized) {
        cout << "[ERROR] CudaKeyFrameStorage::addCudaKeyFrame: ] memory not initialized!\n";
        LoopClosingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }

    if (num_keyframes >= CUDA_KEYFRAME_STORAGE_SIZE) {
        cout << "[ERROR] CudaKeyFrameStorage::addCudaKeyFrame: ] number of keyframes: " << num_keyframes << " is greater than CUDA_KEYFRAME_STORAGE_SIZE: " << CUDA_KEYFRAME_STORAGE_SIZE << "\n";
        LoopClosingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }

    auto it = mnId_to_idx.find(KF->mnId);
    if (it != mnId_to_idx.end()) {
        cout << "CudaKeyFrameStorage::addCudaKeyFrame: ] KF " << KF->mnId << " is already on GPU.\n";
        return &d_keyframes[it->second];        
    }

    // Can we reuse old space?
    ckd_buffer_index_t new_kf_idx = first_free_idx;
    if (!free_idx.empty()) {
        new_kf_idx = free_idx.front();
        free_idx.pop();
    }
    else {
        first_free_idx++;
    }

    h_keyframes[new_kf_idx].setGPUAddress(&d_keyframes[new_kf_idx]);
    h_keyframes[new_kf_idx].setMemory(KF);
    checkCudaError(cudaMemcpy(&d_keyframes[new_kf_idx], &h_keyframes[new_kf_idx], sizeof(LOOP_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "[CudaKeyFrameStorage::] Failed to copy individual element to d_keyframes");

    mnId_to_idx.emplace(KF->mnId, new_kf_idx);
    num_keyframes += 1;

    return &d_keyframes[new_kf_idx];
}


LOOP_DATA_WRAPPER::CudaKeyFrame* LoopClosingCudaKeyFrameStorage::getCudaKeyFrame(long unsigned int mnId){
    auto it = mnId_to_idx.find(mnId);
    if (it != mnId_to_idx.end()) {
        return &d_keyframes[it->second];
    }
    return nullptr;
}


void LoopClosingCudaKeyFrameStorage::shutdown() {
    if (!memory_is_initialized) 
        return;

    for (int i = 0; i < CUDA_KEYFRAME_STORAGE_SIZE; ++i) {
        h_keyframes[i].freeMemory();
    }
    
    cudaFree(d_keyframes);
    cudaFreeHost(h_keyframes);

    memory_is_initialized = false;
}

void LoopClosingCudaKeyFrameStorage::eraseCudaKeyFrame(ORB_SLAM3::KeyFrame* KF) {
    if (!memory_is_initialized) 
        return;

    // std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(KF->mnId);
    if (it == mnId_to_idx.end()) {
        cout << "CudaKeyFrameStorage::eraseCudaKeyFrame: ] KF " << KF->mnId << " not in GPU storage!\n";
        return;        
    }
    ckd_buffer_index_t idx = it->second;

    h_keyframes[idx].setAsEmpty();
    mnId_to_idx.erase(KF->mnId);
    free_idx.push(idx);
    num_keyframes--;
}