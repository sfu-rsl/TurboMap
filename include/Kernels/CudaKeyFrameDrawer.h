#ifndef CUDA_KEYFRAME_DRAWER_H
#define CUDA_KEYFRAME_DRAWER_H

#include <vector>
#include "KeyFrame.h"
#include "CudaUtils.h"
#include "CudaMapPointStorage.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include <mutex>
#include <queue>

#define CUDA_KEYFRAME_DRAWER_STORAGE 500

namespace MAPPING_DATA_WRAPPER {
    class CudaKeyFrame;
}

namespace ORB_SLAM3 {
    class KeyFrame;
    class MapPoint;
}

using ckd_buffer_index_t = int;

class CudaKeyFrameDrawer {
    public:
        static void initializeMemory();
        static MAPPING_DATA_WRAPPER::CudaKeyFrame* getCudaKeyFrame(long unsigned int mnId);
        static MAPPING_DATA_WRAPPER::CudaKeyFrame* addCudaKeyFrame(ORB_SLAM3::KeyFrame* KF);
        static void eraseCudaKeyFrame(ORB_SLAM3::KeyFrame* KF);
        static void updateCudaKeyFrameMapPoint(long unsigned int KF_Id, ORB_SLAM3::MapPoint* mp, int idx);
        static void eraseCudaKeyFrameMapPoint(long unsigned int KF_mnId, int idx);
        static void printDrawerKeyframes();
        static void addFeatureVector(long unsigned int KF_mnId, DBoW2::FeatureVector featVec);
        static void shutdown();
    public:
        static MAPPING_DATA_WRAPPER::CudaKeyFrame *d_keyframes, *h_keyframes;
        static std::unordered_map<long unsigned int, ckd_buffer_index_t> mnId_to_idx; 
        static int num_keyframes;
        static bool memory_is_initialized;
        static ckd_buffer_index_t first_free_idx;
        static std::mutex mtx;
        static std::queue<ckd_buffer_index_t> free_idx;
};

#endif