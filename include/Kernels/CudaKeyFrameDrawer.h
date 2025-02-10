#ifndef CUDA_KEYFRAME_DRAWER_H
#define CUDA_KEYFRAME_DRAWER_H

#include <vector>
#include "KeyFrame.h"
#include "CudaUtils.h"
#include "CudaWrappers/CudaKeyFrame.h"

#define CUDA_KEYFRAME_DRAWER_STORAGE 500

namespace MAPPING_DATA_WRAPPER {
    class CudaKeyFrame;
}

class CudaKeyFrameDrawer {
    public:
        static void initializeMemory();
        static MAPPING_DATA_WRAPPER::CudaKeyFrame* getCudaKeyFrame(long unsigned int mnId);
        static void addCudaKeyFrame(ORB_SLAM3::KeyFrame* KF);
        static void updateCudaKeyFrameMapPoints(ORB_SLAM3::KeyFrame* KF);
        static void eraseCudaKeyFrame(ORB_SLAM3::KeyFrame* KF);
        static void shutdown();
    public:
        static MAPPING_DATA_WRAPPER::CudaKeyFrame *d_keyframes, *h_keyframes;
        // static std::unordered_map<long unsigned int, MAPPING_DATA_WRAPPER::CudaKeyFrame*> id_to_kf; 
        static std::unordered_map<long unsigned int, int> mnId_to_idx; 
        static int num_keyframes;
        static bool memory_is_initialized;
        static int first_free_idx;
};

#endif