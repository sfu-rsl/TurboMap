#include "Kernels/CudaKeyFrameDrawer.h"
#include <csignal> 

MAPPING_DATA_WRAPPER::CudaKeyFrame *CudaKeyFrameDrawer::d_keyframes, *CudaKeyFrameDrawer::h_keyframes;
// std::unordered_map<long unsigned int, MAPPING_DATA_WRAPPER::CudaKeyFrame*> CudaKeyFrameDrawer::id_to_kf;
std::unordered_map<long unsigned int, int> CudaKeyFrameDrawer::mnId_to_idx;
int CudaKeyFrameDrawer::num_keyframes = 0;
bool CudaKeyFrameDrawer::memory_is_initialized = false;
int CudaKeyFrameDrawer::first_free_idx = 0;

void CudaKeyFrameDrawer::initializeMemory(){   
    if (memory_is_initialized) return;
    checkCudaError(cudaMallocHost((void**)&h_keyframes, CUDA_KEYFRAME_DRAWER_STORAGE * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "[CudaKeyFrameDrawer::] Failed to allocate memory for h_keyframes");  
    for (int i = 0; i < CUDA_KEYFRAME_DRAWER_STORAGE; ++i) {
        h_keyframes[i] = MAPPING_DATA_WRAPPER::CudaKeyFrame();
    }

    checkCudaError(cudaMalloc((void**)&d_keyframes, CUDA_KEYFRAME_DRAWER_STORAGE * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "[CudaKeyFrameDrawer::] Failed to allocate memory for d_keyframes");
    memory_is_initialized = true;
}

__global__ void validateKFInput_GPU(MAPPING_DATA_WRAPPER::CudaKeyFrame* KF) {
    printf("***********************CUDA KEYFRAME DRAWER******************************\n");
    printf("PTR: %p\n", (void*)KF);
    printf("MNID: %d\n", KF->mnId);
    for (int i = 0; i < KF->mvpMapPoints_size; ++i) {
        MAPPING_DATA_WRAPPER::CudaMapPoint* mp = KF->mvpMapPoints[i];
        if (mp->isEmpty) {
            printf("[GPU::] i: %d, mp: is empty\n", i);
        } else {
            MAPPING_DATA_WRAPPER::CudaKeyFrame** mObservations_dkf = mp->mObservations_dkf;
            for (int j = 0; j < mp->mObservations_size; ++j) {
                int leftIdx = mp->mObservations_leftIdx[j];
                MAPPING_DATA_WRAPPER::CudaKeyFrame* pKFi = mp->mObservations_dkf[j];
                printf("    [GPU::] j:%d, mp: %lu, pKFi ptr: %p\n", j, mp->mnId, (void*)pKFi);
                printf("    [GPU::] j:%d, mp: %lu, pKFi mnId: %lu\n", j, mp->mnId, (void*)pKFi->mnId);
            }
        }
    }
}

void CudaKeyFrameDrawer::updateCudaKeyFrameMapPoints(ORB_SLAM3::KeyFrame* KF) {
    auto it = mnId_to_idx.find(KF->mnId);
    if (it == mnId_to_idx.end()) {
        cout << "[ERROR] CudaKeyFrameDrawer::modifyCudaKeyFrame: ] KF not found!\n";
        raise(SIGSEGV);        
    }
    int idx = it->second;

    h_keyframes[idx].updateMapPoints(KF->GetMapPointMatches());
    checkCudaError(cudaMemcpy(&d_keyframes[idx], &h_keyframes[idx], sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "[CudaKeyFrameDrawer::modifyCudaKeyFrame: ] Failed to modify keyframe");
}

void CudaKeyFrameDrawer::addCudaKeyFrame(ORB_SLAM3::KeyFrame* KF){
    if (!memory_is_initialized) {
        cout << "[ERROR] CudaKeyFrameDrawer::addCudaKeyFrame: ] memory not initialized!\n";
        raise(SIGSEGV);
    }
    if (num_keyframes > CUDA_KEYFRAME_DRAWER_STORAGE) {
        cout << "[ERROR] CudaKeyFrameDrawer::addCudaKeyFrame: ] number of keyframes: " << num_keyframes << " is greater than CUDA_KEYFRAME_DRAWER_STORAGE: " << CUDA_KEYFRAME_DRAWER_STORAGE << "\n";
        raise(SIGSEGV);
    }

    h_keyframes[first_free_idx].setGPUAddress(&d_keyframes[first_free_idx]);
    h_keyframes[first_free_idx].setMemory(*KF);
    checkCudaError(cudaMemcpy(&d_keyframes[first_free_idx], &h_keyframes[first_free_idx], sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "[CudaKeyFrameDrawer::] Failed to copy individual element to d_keyframes");

    // id_to_kf.emplace(KF->mnId, &d_keyframes[first_free_idx]);
    mnId_to_idx.emplace(KF->mnId, first_free_idx);
    first_free_idx += 1;
    num_keyframes += 1;

    cout << "Drawer added KF: " << KF->mnId << endl;
}

void CudaKeyFrameDrawer::eraseCudaKeyFrame(ORB_SLAM3::KeyFrame* KF){
    auto it = mnId_to_idx.find(KF->mnId);
    if (it == mnId_to_idx.end()) {
        cout << "[ERROR] CudaKeyFrameDrawer::modifyCudaKeyFrame: ] KF not found!\n";
        raise(SIGSEGV);        
    }
    int idx = it->second;

    h_keyframes[idx].freeMemory();
    mnId_to_idx.erase(KF->mnId);
}

MAPPING_DATA_WRAPPER::CudaKeyFrame* CudaKeyFrameDrawer::getCudaKeyFrame(long unsigned int mnId){
    auto it = mnId_to_idx.find(mnId);
    if (it != mnId_to_idx.end()) {
        return &d_keyframes[it->second];
    }
    return nullptr;
}

void CudaKeyFrameDrawer::shutdown() {
    if (!memory_is_initialized) 
    return;

    for (int i = 0; i < CUDA_KEYFRAME_DRAWER_STORAGE; ++i) {
        h_keyframes[i].freeMemory();
    }
    
    cudaFree(d_keyframes);
    cudaFreeHost(h_keyframes);
}
