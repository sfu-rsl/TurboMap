#ifndef CUDA_KEYFRAME_H
#define CUDA_KEYFRAME_H

#include "CudaMapPoint.h"
#include "CudaKeyPoint.h"
#include "KeyFrame.h"
#include "../CudaUtils.h"

namespace MAPPING_DATA_WRAPPER {

class CudaKeyframe {
    private:
        void initializeMemory();

    public:
        CudaKeyframe();
        void setMemory(ORB_SLAM3::KeyFrame* KF);
        void freeMemory();

    public:
        int mnId;
        int Nleft;
        size_t mvKeys_size, mvKeysRight_size, mvKeysUn_size;
        CudaKeyPoint *mvKeys, *mvKeysRight, *mvKeysUn;
        size_t mvpMapPoints_size;
        CudaMapPoint* mvpMapPoints;
    };
}

#endif // CUDA_KEYFRAME_H