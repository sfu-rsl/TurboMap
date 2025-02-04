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
        void setGPUAddress(CudaKeyframe* ptr);
        void setMemory(ORB_SLAM3::KeyFrame* KF);
        void updateMapPoints(vector<ORB_SLAM3::MapPoint*> mvpMapPoints);
        void freeMemory();

    public:
        int mnId;
        int NLeft;
        float mThDepth;
        float* mvDepth;
        size_t mvKeys_size, mvKeysRight_size, mvKeysUn_size;
        CudaKeyPoint *mvKeys, *mvKeysRight, *mvKeysUn;
        size_t mvpMapPoints_size;
        CudaMapPoint** mvpMapPoints;
        CudaKeyframe* gpuAddr;
    };
}

#endif // CUDA_KEYFRAME_H