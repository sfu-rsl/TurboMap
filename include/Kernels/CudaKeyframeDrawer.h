#ifndef CUDA_KEYFRAME_DRAWER_H
#define CUDA_KEYFRAME_DRAWER_H

#include <vector>
#include "KeyFrame.h"
#include "CudaUtils.h"
#include "CudaWrappers/CudaKeyframe.h"
#include <mutex>

#define CUDA_KEYFRAME_DRAWER_STORAGE 500

namespace MAPPING_DATA_WRAPPER {
    class CudaKeyframe;
}

class CudaKeyframeDrawer {
    public:
        static void initializeMemory();
        static MAPPING_DATA_WRAPPER::CudaKeyframe* getCudaKeyframe(long unsigned int mnId);
        static MAPPING_DATA_WRAPPER::CudaKeyframe* addCudaKeyframe(ORB_SLAM3::KeyFrame* KF);
        static void updateCudaKeyframeMapPoint(long unsigned int KF_Id, ORB_SLAM3::MapPoint* mp, int idx);
        static void eraseCudaKeyframe(ORB_SLAM3::KeyFrame* KF);
        static void shutdown();
    public:
        static MAPPING_DATA_WRAPPER::CudaKeyframe *d_keyframes, *h_keyframes;
        // static std::unordered_map<long unsigned int, MAPPING_DATA_WRAPPER::CudaKeyframe*> id_to_kf; 
        static std::unordered_map<long unsigned int, int> mnId_to_idx; 
        static int num_keyframes;
        static bool memory_is_initialized;
        static int first_free_idx;
        static std::mutex mtx;
};

#endif