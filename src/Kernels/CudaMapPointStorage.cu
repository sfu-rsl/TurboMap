#include "Kernels/CudaMapPointStorage.h"
#include <csignal> 

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaMapPointStorage::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

MAPPING_DATA_WRAPPER::CudaMapPoint *CudaMapPointStorage::d_mappoints, *CudaMapPointStorage::h_mappoints;
std::unordered_map<long unsigned int, MAPPING_DATA_WRAPPER::CudaMapPoint*> CudaMapPointStorage::id_to_mp;
std::unordered_map<long unsigned int, int> CudaMapPointStorage::mnId_to_idx;
int CudaMapPointStorage::num_mappoints = 0;
bool CudaMapPointStorage::memory_is_initialized = false;
int CudaMapPointStorage::first_free_idx = 0;

void CudaMapPointStorage::initializeMemory(){   
    if (memory_is_initialized) return;
    checkCudaError(cudaMallocHost((void**)&h_mappoints, CUDA_MAP_POINT_STORAGE_SIZE * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint)), "[CudaMapPointStorage::] Failed to allocate memory for h_mappoints");  
    // for (int i = 0; i < CUDA_MAP_POINT_STORAGE_SIZE; ++i) {
    //     h_mappoints[i] = MAPPING_DATA_WRAPPER::CudaMapPoint();
    // }
    checkCudaError(cudaMalloc((void**)&d_mappoints, CUDA_MAP_POINT_STORAGE_SIZE * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint)), "[CudaMapPointStorage::] Failed to allocate memory for d_mappoints");
    memory_is_initialized = true;
}

MAPPING_DATA_WRAPPER::CudaMapPoint* CudaMapPointStorage::modifyCudaMapPoint(long unsigned int mnId, ORB_SLAM3::MapPoint* new_MP) {
    int idx;
    auto it = mnId_to_idx.find(mnId);
    if (it == mnId_to_idx.end()) {
        DEBUG_PRINT("::modifyCudaMapPoint: " << mnId << " not found! Adding mp to storage.." << endl);
        return CudaMapPointStorage::addCudaMapPoint(new_MP);     
    }
    idx = it->second;

    h_mappoints[idx] = MAPPING_DATA_WRAPPER::CudaMapPoint(new_MP);
    checkCudaError(cudaMemcpy(&d_mappoints[idx], &h_mappoints[idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage:: Modify Map Point] Failed to modify mappoint");
    mnId_to_idx.erase(mnId);
    mnId_to_idx.emplace(new_MP->mnId, idx);

    DEBUG_PRINT("modifyCudaMapPoint: " << mnId << " (mappoints on gpu: " << num_mappoints << ")" << endl);

    return &d_mappoints[idx];
}

MAPPING_DATA_WRAPPER::CudaMapPoint* CudaMapPointStorage::addCudaMapPoint(ORB_SLAM3::MapPoint* MP){
    if (!memory_is_initialized) {
        cout << "[ERROR] CudaMapPointStorage::addCudaMapPoint: ] memory not initialized!\n";
        raise(SIGSEGV);
    }

    if (num_mappoints > CUDA_MAP_POINT_STORAGE_SIZE) {
        cout << "[ERROR] CudaMapPointStorage::addCudaMapPoint: ] number of mappoints: " << num_mappoints << " is greater than CUDA_MAP_POINT_STORAGE_SIZE: " << CUDA_MAP_POINT_STORAGE_SIZE << "\n";
        raise(SIGSEGV);
    }

    h_mappoints[first_free_idx] = MAPPING_DATA_WRAPPER::CudaMapPoint(MP);
    checkCudaError(cudaMemcpy(&d_mappoints[first_free_idx], &h_mappoints[first_free_idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage::] Failed to copy individual element to d_mappoints");

    // id_to_mp.emplace(MP->mnId, &d_mappoints[first_free_idx]);
    mnId_to_idx.emplace(MP->mnId, first_free_idx);
    first_free_idx += 1;
    num_mappoints += 1;

    DEBUG_PRINT("addCudaMapPoint: " << MP->mnId << " (mappoints on gpu: " << num_mappoints << ")" << endl);

    return &d_mappoints[first_free_idx-1];
}

MAPPING_DATA_WRAPPER::CudaMapPoint* CudaMapPointStorage::keepCudaMapPoint(MAPPING_DATA_WRAPPER::CudaMapPoint cuda_mp){
    if (!memory_is_initialized) {
        cout << "[ERROR] CudaMapPointStorage::keepCudaMapPoint: ] memory not initialized!\n";
        raise(SIGSEGV);
    }

    if (num_mappoints > CUDA_MAP_POINT_STORAGE_SIZE) {
        cout << "[ERROR] CudaMapPointStorage::keepCudaMapPoint: ] number of mappoints: " << num_mappoints << " is greater than CUDA_MAP_POINT_STORAGE_SIZE: " << CUDA_MAP_POINT_STORAGE_SIZE << "\n";
        raise(SIGSEGV);
    }

    h_mappoints[first_free_idx] = cuda_mp;
    checkCudaError(cudaMemcpy(&d_mappoints[first_free_idx], &h_mappoints[first_free_idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage::] Failed to copy individual element to d_mappoints");
    // id_to_mp.emplace(MP->mnId, &d_mappoints[first_free_idx]);
    mnId_to_idx.emplace(cuda_mp.mnId, first_free_idx);
    first_free_idx += 1;
    num_mappoints += 1;

    DEBUG_PRINT("keepCudaMapPoint: " << cuda_mp.mnId << " (mappoints on gpu: " << num_mappoints << ")" << endl);

    return &d_mappoints[first_free_idx-1];
}

void CudaMapPointStorage::eraseCudaMapPoint(ORB_SLAM3::MapPoint* MP){
    auto it = mnId_to_idx.find(MP->mnId);
    if (it == mnId_to_idx.end()) {
        cout << "[ERROR] CudaMapPointStorage::eraseCudaMapPoint: ] mp not found!\n";
        return;       
    }
    int idx = it->second;

    // h_mappoints[idx] = MAPPING_DATA_WRAPPER::CudaMapPoint();
    // checkCudaError(cudaMemcpy(&d_mappoints[idx], &h_mappoints[idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage:: Erase Map Point] Failed to erase mappoint");
    mnId_to_idx.erase(MP->mnId);

    DEBUG_PRINT("eraseCudaMapPoint: " << MP->mnId << " (mappoints on gpu: " << num_mappoints << ")" << endl);
}

MAPPING_DATA_WRAPPER::CudaMapPoint* CudaMapPointStorage::getCudaMapPoint(long unsigned int mnId){
    auto it = mnId_to_idx.find(mnId);
    if (it != mnId_to_idx.end()) {
        return &d_mappoints[it->second];
    }
    return nullptr;
}


void CudaMapPointStorage::shutdown() {
    if (!memory_is_initialized) 
        return;
    cudaFree(h_mappoints);
    cudaFreeHost(d_mappoints);
}