#include "Kernels/CudaMapPointStorage.h"
#include "Kernels/MappingKernelController.h"
#include "Stats/LocalMappingStats.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaMapPointStorage::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

MAPPING_DATA_WRAPPER::CudaMapPoint *CudaMapPointStorage::d_mappoints, *CudaMapPointStorage::h_mappoints;
std::unordered_map<long unsigned int, cmp_buffer_index_t> CudaMapPointStorage::mnId_to_idx;
int CudaMapPointStorage::num_mappoints = 0;
bool CudaMapPointStorage::memory_is_initialized = false;
cmp_buffer_index_t CudaMapPointStorage::first_free_idx = 0;
std::mutex CudaMapPointStorage::mtx;
std::queue<cmp_buffer_index_t> CudaMapPointStorage::free_idx;


void CudaMapPointStorage::initializeMemory() {   
    // std::unique_lock<std::mutex> lock(mtx);
    if (memory_is_initialized) return;
    checkCudaError(cudaMallocHost((void**)&h_mappoints, CUDA_MAP_POINT_STORAGE_SIZE * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint)), "[CudaMapPointStorage::] Failed to allocate memory for h_mappoints");  
    for (int i = 0; i < CUDA_MAP_POINT_STORAGE_SIZE; ++i) {
        h_mappoints[i] = MAPPING_DATA_WRAPPER::CudaMapPoint();
    }
    checkCudaError(cudaMalloc((void**)&d_mappoints, CUDA_MAP_POINT_STORAGE_SIZE * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint)), "[CudaMapPointStorage::] Failed to allocate memory for d_mappoints");
    memory_is_initialized = true;
}

MAPPING_DATA_WRAPPER::CudaMapPoint* CudaMapPointStorage::replaceCudaMapPoint(long unsigned int mnId, ORB_SLAM3::MapPoint* new_MP) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif  
    // mtx.lock();
    int idx;
    auto it = mnId_to_idx.find(mnId);
    if (it == mnId_to_idx.end()) {
        mtx.unlock(); // TODO: avoid unlock + relock in addCudaMapPoint
        DEBUG_PRINT("::replaceCudaMapPoint: " << mnId << " not found! Adding mp to storage.." << endl);
        return CudaMapPointStorage::addCudaMapPoint(new_MP);     
    }
    idx = it->second;

    h_mappoints[idx].setMemory(new_MP);
    checkCudaError(cudaMemcpy(&d_mappoints[idx], &h_mappoints[idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage:: Modify Map Point] Failed to replace mappoint");
    mnId_to_idx.erase(mnId);
    mnId_to_idx.emplace(new_MP->mnId, idx);

    DEBUG_PRINT("replaceCudaMapPoint: " << mnId << " (mappoints on gpu: " << num_mappoints << ")" << endl);
    
    auto ret = &d_mappoints[idx];
    // mtx.unlock();
    // return &d_mappoints[idx];
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().replaceCudaMapPoint_time.push_back(time);
#endif
    return ret;
}

void CudaMapPointStorage::updateCudaMapPointObservations(long unsigned int mnId, int nObs, map<ORB_SLAM3::KeyFrame*, tuple<int,int>> observations) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif 
    // mtx.lock();
    int idx;
    auto it = mnId_to_idx.find(mnId);
    if (it == mnId_to_idx.end()) {
        // mtx.unlock(); // TODO: avoid unlock + relock in addCudaMapPoint
        cout << "[ERROR] CudaMapPointStorage::updateCudaMapPointObservations: ] mp not in GPU storage!\n";
        return;    
    }
    idx = it->second;

    h_mappoints[idx].setObservations(nObs, observations);
    checkCudaError(cudaMemcpy(&d_mappoints[idx], &h_mappoints[idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage:: Modify Map Point] Failed to update mappoint");

    DEBUG_PRINT("updateCudaMapPointObservations: " << mnId << endl);
    // mtx.unlock();
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().updateCudaMapPointObservations_time.push_back(time);
#endif

    return;
}

void CudaMapPointStorage::updateCudaMapPointWorldPos(long unsigned int mnId, Eigen::Vector3f Pos) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif
    // mtx.lock();
    int idx;
    auto it = mnId_to_idx.find(mnId);
    if (it == mnId_to_idx.end()) {
        // mtx.unlock(); // TODO: avoid unlock + relock in addCudaMapPoint
        cout << "[ERROR] CudaMapPointStorage::updateCudaMapPointWorldPos: ] mp not in GPU storage!\n";
        return;    
    }
    idx = it->second;

    h_mappoints[idx].setWorldPos(Pos);
    checkCudaError(cudaMemcpy(&d_mappoints[idx], &h_mappoints[idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage:: Modify Map Point] Failed to update mappoint world pos");

    DEBUG_PRINT("updateCudaMapPointWorldPos: " << mnId << endl);
    // mtx.unlock();

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().updateCudaMapPointWorldPos_time.push_back(time);
#endif
} 

void CudaMapPointStorage::updateCudaMapNormalAndDepth(long unsigned int mnId, float mfMinDistance, float mfMaxDistance, Eigen::Vector3f mNormalVector) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif
    // mtx.lock();
    int idx;
    auto it = mnId_to_idx.find(mnId);
    if (it == mnId_to_idx.end()) {
        // mtx.unlock(); // TODO: avoid unlock + relock in addCudaMapPoint
        cout << "[ERROR] CudaMapPointStorage::updateCudaMapNormalAndDepth: ] mp not in GPU storage!\n";
        return;    
    }
    idx = it->second;

    h_mappoints[idx].setMinDistance(mfMinDistance);
    h_mappoints[idx].setMaxDistance(mfMaxDistance);
    h_mappoints[idx].setNormalVector(mNormalVector);
    checkCudaError(cudaMemcpy(&d_mappoints[idx], &h_mappoints[idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage:: Modify Map Point] Failed to update mappoint normal and depth");

    DEBUG_PRINT("updateCudaMapNormalAndDepth: " << mnId << endl);
    // mtx.unlock();

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().updateCudaMapNormalAndDepth_time.push_back(time);
#endif
}

void CudaMapPointStorage::updateCudaMapPointDescriptor(long unsigned int mnId, cv::Mat mDescriptor) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif
    // mtx.lock();
    int idx;
    auto it = mnId_to_idx.find(mnId);
    if (it == mnId_to_idx.end()) {
        // mtx.unlock(); // TODO: avoid unlock + relock in addCudaMapPoint
        cout << "[ERROR] CudaMapPointStorage::updateCudaMapPointDescriptor: ] mp not in GPU storage!\n";
        return;    
    }
    idx = it->second;

    h_mappoints[idx].setDescriptor(mDescriptor);
    checkCudaError(cudaMemcpy(&d_mappoints[idx], &h_mappoints[idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage:: Modify Map Point] Failed to modify mappoint");

    DEBUG_PRINT("updateCudaMapPointDescriptor: " << mnId << endl);
    // mtx.unlock();

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().updateCudaMapPointDescriptor_time.push_back(time);
#endif
}

MAPPING_DATA_WRAPPER::CudaMapPoint* CudaMapPointStorage::addCudaMapPoint(ORB_SLAM3::MapPoint* MP) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif
    // std::unique_lock<std::mutex> lock(mtx);
    if (!memory_is_initialized) {
        cout << "[ERROR] CudaMapPointStorage::addCudaMapPoint: ] memory not initialized!\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }

    if (num_mappoints >= CUDA_MAP_POINT_STORAGE_SIZE) {
        cout << "[ERROR] CudaMapPointStorage::addCudaMapPoint: ] number of mappoints: " << num_mappoints << " is greater than CUDA_MAP_POINT_STORAGE_SIZE: " << CUDA_MAP_POINT_STORAGE_SIZE << "\n";
        MappingKernelController::shutdownKernels(true, true);
        exit(EXIT_FAILURE);
    }

    auto it = mnId_to_idx.find(MP->mnId);
    if (it != mnId_to_idx.end()) {
        cout << "CudaMapPointStorage::addCudaMapPoint: ] MP " << MP->mnId << " is already on GPU.\n";
        return &d_mappoints[it->second];        
    }

    // Reuse available space if possible
    cmp_buffer_index_t new_mp_idx = first_free_idx;
    if (!free_idx.empty()) {
        new_mp_idx = free_idx.front();
        free_idx.pop();
    }
    else {
        first_free_idx++;
    }

    h_mappoints[new_mp_idx].setMemory(MP);
    checkCudaError(cudaMemcpy(&d_mappoints[new_mp_idx], &h_mappoints[new_mp_idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage::] Failed to copy individual element to d_mappoints");

    mnId_to_idx.emplace(MP->mnId, new_mp_idx);
    num_mappoints += 1;

    DEBUG_PRINT("addCudaMapPoint: " << MP->mnId << " (mappoints on gpu: " << num_mappoints << ")" << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().addCudaMapPoint_time.push_back(time);
#endif

    return &d_mappoints[new_mp_idx];
}

void CudaMapPointStorage::eraseCudaMapPoint(ORB_SLAM3::MapPoint* MP) {
#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif
    // std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(MP->mnId);
    if (it == mnId_to_idx.end()) {
        cout << "[ERROR] CudaMapPointStorage::eraseCudaMapPoint: ] mp not in GPU storage!\n";
        return;       
    }
    cmp_buffer_index_t idx = it->second;

    h_mappoints[idx].setAsEmpty();
    checkCudaError(cudaMemcpy(&d_mappoints[idx], &h_mappoints[idx], sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[CudaMapPointStorage:: Erase Map Point] Failed to erase mappoint");
    mnId_to_idx.erase(MP->mnId);
    free_idx.push(idx);
    num_mappoints--;

    DEBUG_PRINT("eraseCudaMapPoint: " << MP->mnId << " (mappoints on gpu: " << num_mappoints << ")" << endl);

#ifdef REGISTER_LOCAL_MAPPING_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    LocalMappingStats::getInstance().eraseCudaMapPoint_time.push_back(time);
#endif
}

MAPPING_DATA_WRAPPER::CudaMapPoint* CudaMapPointStorage::getCudaMapPoint(long unsigned int mnId) {
    // std::unique_lock<std::mutex> lock(mtx);
    auto it = mnId_to_idx.find(mnId);
    if (it != mnId_to_idx.end()) {
        return &d_mappoints[it->second];
    }
    return nullptr;
}

void CudaMapPointStorage::printStorageMapPoints() {
    cout << "[";
    for (const auto& pair : mnId_to_idx) {
        std::cout << pair.first << ", ";
    }
    cout << "]\n";
}

void CudaMapPointStorage::shutdown() {
    // std::unique_lock<std::mutex> lock(mtx);
    if (!memory_is_initialized) 
        return;
    cout << "CudaMapPointStorage::shutdown() start" << endl;
    for (int i = 0; i < CUDA_MAP_POINT_STORAGE_SIZE; ++i) {
        h_mappoints[i].freeMemory();
    }
    cudaFreeHost(h_mappoints);
    cudaFree(d_mappoints);
    cout << "CudaMapPointStorage::shutdown() end" << endl;
}