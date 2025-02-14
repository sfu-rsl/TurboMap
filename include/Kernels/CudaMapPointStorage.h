#ifndef CUDA_MAP_POINT_STORAGE
#define CUDA_MAP_POINT_STORAGE

#include <vector>
#include "../MapPoint.h"
#include "CudaUtils.h"
#include "CudaWrappers/CudaMapPoint.h"

#define CUDA_MAP_POINT_STORAGE_SIZE 50000

class CudaMapPointStorage {
    public:
        static void initializeMemory();
        static MAPPING_DATA_WRAPPER::CudaMapPoint* getCudaMapPoint(long unsigned int mnId);
        static MAPPING_DATA_WRAPPER::CudaMapPoint* modifyCudaMapPoint(long unsigned int mnId, ORB_SLAM3::MapPoint* new_MP); 
        static MAPPING_DATA_WRAPPER::CudaMapPoint* addCudaMapPoint(ORB_SLAM3::MapPoint* MP);
        static void eraseCudaMapPoint(ORB_SLAM3::MapPoint* MP);
        static MAPPING_DATA_WRAPPER::CudaMapPoint* keepCudaMapPoint(MAPPING_DATA_WRAPPER::CudaMapPoint cuda_mp);
        static void shutdown();
    public:
        static MAPPING_DATA_WRAPPER::CudaMapPoint *d_mappoints, *h_mappoints;
        static std::unordered_map<long unsigned int, MAPPING_DATA_WRAPPER::CudaMapPoint*> id_to_mp; 
        static std::unordered_map<long unsigned int, int> mnId_to_idx; 
        static int num_mappoints;
        static bool memory_is_initialized;
        static int first_free_idx;
};

#endif