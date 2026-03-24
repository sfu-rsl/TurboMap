#ifndef LOOP_CLOSING_CUDA_KEYFRAME_STORAGE_H
#define LOOP_CLOSING_CUDA_KEYFRAME_STORAGE_H


#include <vector>
#include "KeyFrame.h"
#include "CudaUtils.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include <mutex>
#include <queue>

#define CUDA_KEYFRAME_STORAGE_SIZE 1000

using ckd_buffer_index_t = int;


class LoopClosingCudaKeyFrameStorage {
    public:
        static void initializeMemory();
        static LOOP_DATA_WRAPPER::CudaKeyFrame* addCudaKeyFrame(ORB_SLAM3::KeyFrame* KF);
        static LOOP_DATA_WRAPPER::CudaKeyFrame* getCudaKeyFrame(long unsigned int mnId);
        static void eraseCudaKeyFrame(ORB_SLAM3::KeyFrame* KF);
        static void shutdown();

    public:
        static LOOP_DATA_WRAPPER::CudaKeyFrame *d_keyframes, *h_keyframes;
        static int num_keyframes;
        static bool memory_is_initialized;
        static ckd_buffer_index_t first_free_idx;
        static std::queue<ckd_buffer_index_t> free_idx;
        static std::unordered_map<long unsigned int, ckd_buffer_index_t> mnId_to_idx;
        static std::mutex mtx;

};

#endif