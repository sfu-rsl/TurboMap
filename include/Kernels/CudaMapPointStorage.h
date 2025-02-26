#ifndef CUDA_MAP_POINT_STORAGE
#define CUDA_MAP_POINT_STORAGE

#include <vector>
#include "../MapPoint.h"
#include "CudaUtils.h"
#include "CudaWrappers/CudaMapPoint.h"
#include <queue>
#include <mutex>

#define CUDA_MAP_POINT_STORAGE_SIZE 500000

using cmp_buffer_index_t = int;

namespace ORB_SLAM3 {
    class MapPoint;
}

class CudaMapPointStorage {
    public:
        static void initializeMemory();
        static MAPPING_DATA_WRAPPER::CudaMapPoint* getCudaMapPoint(long unsigned int mnId);
        static MAPPING_DATA_WRAPPER::CudaMapPoint* replaceCudaMapPoint(long unsigned int mnId, ORB_SLAM3::MapPoint* new_MP); 
        static void setCudaMapPointObservations(long unsigned int mnId, int nObs, map<ORB_SLAM3::KeyFrame*, tuple<int,int>> observations);
        static MAPPING_DATA_WRAPPER::CudaMapPoint* addCudaMapPoint(ORB_SLAM3::MapPoint* MP);
        static void eraseCudaMapPoint(ORB_SLAM3::MapPoint* MP);
        static void printStorageMapPoints();
        static void shutdown();
    public:
        static MAPPING_DATA_WRAPPER::CudaMapPoint *d_mappoints, *h_mappoints; 
        static std::unordered_map<long unsigned int, cmp_buffer_index_t> mnId_to_idx; 
        static int num_mappoints;
        static bool memory_is_initialized;
        static cmp_buffer_index_t first_free_idx;
        static std::mutex mtx;
        static std::queue<cmp_buffer_index_t> free_idx;
};

#endif