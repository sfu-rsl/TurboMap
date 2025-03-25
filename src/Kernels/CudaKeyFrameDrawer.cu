#include "Kernels/CudaKeyFrameDrawer.h"
#include "Kernels/MappingKernelController.h"
#include <csignal> 

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaKeyFrameDrawer::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

MAPPING_DATA_WRAPPER::CudaKeyFrame *CudaKeyFrameDrawer::d_keyframes, *CudaKeyFrameDrawer::h_keyframes;
std::unordered_map<long unsigned int, ckd_buffer_index_t> CudaKeyFrameDrawer::mnId_to_idx;
int CudaKeyFrameDrawer::num_keyframes = 0;
bool CudaKeyFrameDrawer::memory_is_initialized = false;
ckd_buffer_index_t CudaKeyFrameDrawer::first_free_idx = 0;
std::mutex CudaKeyFrameDrawer::mtx;
std::queue<ckd_buffer_index_t> CudaKeyFrameDrawer::free_idx;


void CudaKeyFrameDrawer::initializeMemory(){   
    std::unique_lock<std::mutex> lock(mtx);
    if (memory_is_initialized) return;
    checkCudaError(cudaMallocHost((void**)&h_keyframes, CUDA_KEYFRAME_DRAWER_STORAGE * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "[CudaKeyFrameDrawer::] Failed to allocate memory for h_keyframes");  
    for (int i = 0; i < CUDA_KEYFRAME_DRAWER_STORAGE; ++i) {
        h_keyframes[i] = MAPPING_DATA_WRAPPER::CudaKeyFrame();
    }

    checkCudaError(cudaMalloc((void**)&d_keyframes, CUDA_KEYFRAME_DRAWER_STORAGE * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "[CudaKeyFrameDrawer::] Failed to allocate memory for d_keyframes");
    memory_is_initialized = true;
}

void CudaKeyFrameDrawer::eraseCudaKeyFrameMapPoint(long unsigned int KF_mnId, int idx) {
    std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(KF_mnId);
    if (it == mnId_to_idx.end()) {
        cout << "[ERROR] CudaKeyFrameDrawer::eraseCudaKeyFrameMapPoint: ] KF " << KF_mnId << " not found!\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }
    int KF_idx = it->second;

    h_keyframes[KF_idx].eraseMapPoint(idx);
    checkCudaError(cudaMemcpy(&d_keyframes[KF_idx], &h_keyframes[KF_idx], sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "[CudaKeyFrameDrawer::eraseCudaKeyFrameMapPoint: ] Failed");

    DEBUG_PRINT("eraseCudaKeyFrameMapPoint: " << KF_mnId << endl);

}

void CudaKeyFrameDrawer::updateCudaKeyFrameMapPoint(long unsigned int KF_mnId, ORB_SLAM3::MapPoint* mp, int idx) {
    std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(KF_mnId);
    if (it == mnId_to_idx.end()) {
        cout << "[ERROR] CudaKeyFrameDrawer::updateCudaKeyFrameMapPoint: ] KF " << KF_mnId << " not found!\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);       
    }
    int KF_idx = it->second;

    MAPPING_DATA_WRAPPER::CudaMapPoint* d_mp = CudaMapPointStorage::getCudaMapPoint(mp->mnId);
    h_keyframes[KF_idx].addMapPoint(d_mp, idx);
    checkCudaError(cudaMemcpy(&d_keyframes[KF_idx], &h_keyframes[KF_idx], sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "[CudaKeyFrameDrawer::updateCudaKeyFrameMapPoint: ] Failed");

    DEBUG_PRINT("updateCudaKeyFrameMapPoint: " << KF_mnId << endl);
}

MAPPING_DATA_WRAPPER::CudaKeyFrame* CudaKeyFrameDrawer::addCudaKeyFrame(ORB_SLAM3::KeyFrame* KF){
    std::unique_lock<std::mutex> lock(mtx);
    if (!memory_is_initialized) {
        cout << "[ERROR] CudaKeyFrameDrawer::addCudaKeyFrame: ] memory not initialized!\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }
    if (num_keyframes >= CUDA_KEYFRAME_DRAWER_STORAGE) {
        cout << "[ERROR] CudaKeyFrameDrawer::addCudaKeyFrame: ] number of keyframes: " << num_keyframes << " is greater than CUDA_KEYFRAME_DRAWER_STORAGE: " << CUDA_KEYFRAME_DRAWER_STORAGE << "\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }

    auto it = mnId_to_idx.find(KF->mnId);
    if (it != mnId_to_idx.end()) {
        cout << "CudaKeyFrameDrawer::addCudaKeyFrame: ] KF " << KF->mnId << " is already on GPU.\n";
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
    checkCudaError(cudaMemcpy(&d_keyframes[new_kf_idx], &h_keyframes[new_kf_idx], sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "[CudaKeyFrameDrawer::] Failed to copy individual element to d_keyframes");

    mnId_to_idx.emplace(KF->mnId, new_kf_idx);
    num_keyframes += 1;

    DEBUG_PRINT("addCudaKeyFrame: " << KF->mnId << endl);

    return &d_keyframes[new_kf_idx];
}

void CudaKeyFrameDrawer::eraseCudaKeyFrame(ORB_SLAM3::KeyFrame* KF){
    std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(KF->mnId);
    if (it == mnId_to_idx.end()) {
        cout << "CudaKeyFrameDrawer::eraseCudaKeyFrame: ] KF " << KF->mnId << " not in GPU storage!\n";
        return;        
    }
    ckd_buffer_index_t idx = it->second;

    h_keyframes[idx].setAsEmpty();
    mnId_to_idx.erase(KF->mnId);
    free_idx.push(idx);
    num_keyframes--;

    DEBUG_PRINT("eraseCudaKeyFrame: " << KF->mnId << endl);
}

MAPPING_DATA_WRAPPER::CudaKeyFrame* CudaKeyFrameDrawer::getCudaKeyFrame(long unsigned int mnId){
    std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(mnId);
    if (it != mnId_to_idx.end()) {
        return &d_keyframes[it->second];
    }
    return nullptr;
}

void CudaKeyFrameDrawer::printDrawerKeyframes() {
    cout << "[";
    for (const auto& pair : mnId_to_idx) {
        std::cout << pair.first << ", ";
    }
    cout << "]\n";
}

void CudaKeyFrameDrawer::addFeatureVector(long unsigned int KF_mnId, DBoW2::FeatureVector featVec) {
    std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(KF_mnId);
    if (it == mnId_to_idx.end()) {
        cout << "[ERROR] CudaKeyFrameDrawer::addFeatureVector: ] KF not found!\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);        
    }
    int KF_idx = it->second;
    
    h_keyframes[KF_idx].addFeatureVector(featVec);
    checkCudaError(cudaMemcpy(&d_keyframes[KF_idx], &h_keyframes[KF_idx], sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "[CudaKeyFrameDrawer::addFeatureVector: ] Failed to add feature vector");

    DEBUG_PRINT("addFeatureVector: " << KF_mnId << endl);
}

void CudaKeyFrameDrawer::shutdown() {
    std::unique_lock<std::mutex> lock(mtx);
    if (!memory_is_initialized) 
    return;

    for (int i = 0; i < CUDA_KEYFRAME_DRAWER_STORAGE; ++i) {
        h_keyframes[i].freeMemory();
    }
    
    cudaFree(d_keyframes);
    cudaFreeHost(h_keyframes);
}
