#include <vector>
#include "KeyFrame.h"
#include "CudaUtils.h"
#include "CudaWrappers/CudaKeyframe.h"

#define CUDA_KEYFRAME_DRAWER_STORAGE 1000

namespace MAPPING_DATA_WRAPPER {
    class CudaKeyframe;
}

class CudaKeyframeDrawer {
    public:
        static MAPPING_DATA_WRAPPER::CudaKeyframe* getCudaKeyframe(long unsigned int mnId);
        static MAPPING_DATA_WRAPPER::CudaKeyframe* addCudaKeyframe(ORB_SLAM3::KeyFrame* KF);
    public:
        static MAPPING_DATA_WRAPPER::CudaKeyframe *d_keyframes, *h_keyframes;
        static std::unordered_map<long unsigned int, MAPPING_DATA_WRAPPER::CudaKeyframe*> id_to_kf; 
        static int num_keyframes;
    private:
        static void initializeMemory();
        static bool memory_is_initialized;
};