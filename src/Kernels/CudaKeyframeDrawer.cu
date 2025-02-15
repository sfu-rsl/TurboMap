#include "Kernels/CudaKeyframeDrawer.h"
#include <csignal> 

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaKeyframeDrawer::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

MAPPING_DATA_WRAPPER::CudaKeyframe *CudaKeyframeDrawer::d_keyframes, *CudaKeyframeDrawer::h_keyframes;
// std::unordered_map<long unsigned int, MAPPING_DATA_WRAPPER::CudaKeyframe*> CudaKeyframeDrawer::id_to_kf;
std::unordered_map<long unsigned int, ckd_buffer_index_t> CudaKeyframeDrawer::mnId_to_idx;
int CudaKeyframeDrawer::num_keyframes = 0;
bool CudaKeyframeDrawer::memory_is_initialized = false;
ckd_buffer_index_t CudaKeyframeDrawer::first_free_idx = 0;
std::mutex CudaKeyframeDrawer::mtx;
std::queue<ckd_buffer_index_t> CudaKeyframeDrawer::free_idx;


void CudaKeyframeDrawer::initializeMemory(){   
    std::unique_lock<std::mutex> lock(mtx);
    if (memory_is_initialized) return;
    checkCudaError(cudaMallocHost((void**)&h_keyframes, CUDA_KEYFRAME_DRAWER_STORAGE * sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe)), "[CudaKeyframeDrawer::] Failed to allocate memory for h_keyframes");  
    for (int i = 0; i < CUDA_KEYFRAME_DRAWER_STORAGE; ++i) {
        h_keyframes[i] = MAPPING_DATA_WRAPPER::CudaKeyframe();
    }

    checkCudaError(cudaMalloc((void**)&d_keyframes, CUDA_KEYFRAME_DRAWER_STORAGE * sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe)), "[CudaKeyframeDrawer::] Failed to allocate memory for d_keyframes");
    memory_is_initialized = true;
}

__global__ void validateKFInput_GPU(MAPPING_DATA_WRAPPER::CudaKeyframe* KF) {
    printf("***********************CUDA KEYFRAME DRAWER******************************\n");
    printf("PTR: %p\n", (void*)KF);
    printf("MNID: %d\n", KF->mnId);
    for (int i = 0; i < KF->mvpMapPoints_size; ++i) {
        MAPPING_DATA_WRAPPER::CudaMapPoint* mp = KF->mvpMapPoints[i];
        if (mp->isEmpty) {
            printf("[GPU::] i: %d, mp: is empty\n", i);
        } else {
            MAPPING_DATA_WRAPPER::CudaKeyframe** mObservations_dkf = mp->mObservations_dkf;
            for (int j = 0; j < mp->mObservations_size; ++j) {
                int leftIdx = mp->mObservations_leftIdx[j];
                MAPPING_DATA_WRAPPER::CudaKeyframe* pKFi = mp->mObservations_dkf[j];
                printf("    [GPU::] j:%d, mp: %lu, pKFi ptr: %p\n", j, mp->mnId, (void*)pKFi);
                printf("    [GPU::] j:%d, mp: %lu, pKFi mnId: %lu\n", j, mp->mnId, (void*)pKFi->mnId);
            }
        }
    }
}

void CudaKeyframeDrawer::updateCudaKeyframeMapPoint(long unsigned int KF_mnId, ORB_SLAM3::MapPoint* mp, int idx) {
    std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(KF_mnId);
    if (it == mnId_to_idx.end()) {
        cout << "[ERROR] CudaKeyframeDrawer::modifyCudaKeyframe: ] KF not found!\n";
        raise(SIGSEGV);        
    }
    int KF_idx = it->second;

    h_keyframes[KF_idx].addMapPoint(mp, idx);
    checkCudaError(cudaMemcpy(&d_keyframes[KF_idx], &h_keyframes[KF_idx], sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe), cudaMemcpyHostToDevice), "[CudaKeyframeDrawer::modifyCudaKeyframe: ] Failed to modify keyframe");

    DEBUG_PRINT("updateCudaKeyframeMapPoint: " << KF_mnId << endl);
}

MAPPING_DATA_WRAPPER::CudaKeyframe* CudaKeyframeDrawer::addCudaKeyframe(ORB_SLAM3::KeyFrame* KF){
    std::unique_lock<std::mutex> lock(mtx);
    if (!memory_is_initialized) {
        cout << "[ERROR] CudaKeyframeDrawer::addCudaKeyframe: ] memory not initialized!\n";
        raise(SIGSEGV);
    }
    if (num_keyframes >= CUDA_KEYFRAME_DRAWER_STORAGE) {
        cout << "[ERROR] CudaKeyframeDrawer::addCudaKeyframe: ] number of keyframes: " << num_keyframes << " is greater than CUDA_KEYFRAME_DRAWER_STORAGE: " << CUDA_KEYFRAME_DRAWER_STORAGE << "\n";
        raise(SIGSEGV);
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
    checkCudaError(cudaMemcpy(&d_keyframes[new_kf_idx], &h_keyframes[new_kf_idx], sizeof(MAPPING_DATA_WRAPPER::CudaKeyframe), cudaMemcpyHostToDevice), "[CudaKeyframeDrawer::] Failed to copy individual element to d_keyframes");

    mnId_to_idx.emplace(KF->mnId, new_kf_idx);
    num_keyframes += 1;

    DEBUG_PRINT("addCudaKeyframe: " << KF->mnId << endl);

    return &d_keyframes[new_kf_idx];
}

void CudaKeyframeDrawer::eraseCudaKeyframe(ORB_SLAM3::KeyFrame* KF){
    std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(KF->mnId);
    if (it == mnId_to_idx.end()) {
        cout << "[ERROR] CudaKeyframeDrawer::modifyCudaKeyframe: ] KF not found!\n";
        raise(SIGSEGV);        
    }
    ckd_buffer_index_t idx = it->second;

    h_keyframes[idx].freeMemory();
    mnId_to_idx.erase(KF->mnId);
    free_idx.push(idx);
    num_keyframes--;
    DEBUG_PRINT("eraseCudaKeyframe: " << KF->mnId << endl);
}

MAPPING_DATA_WRAPPER::CudaKeyframe* CudaKeyframeDrawer::getCudaKeyframe(long unsigned int mnId){
    std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(mnId);
    if (it != mnId_to_idx.end()) {
        return &d_keyframes[it->second];
    }
    return nullptr;
}

void CudaKeyframeDrawer::shutdown() {
    std::unique_lock<std::mutex> lock(mtx);
    if (!memory_is_initialized) 
    return;

    for (int i = 0; i < CUDA_KEYFRAME_DRAWER_STORAGE; ++i) {
        h_keyframes[i].freeMemory();
    }
    
    cudaFree(d_keyframes);
    cudaFreeHost(h_keyframes);
}
