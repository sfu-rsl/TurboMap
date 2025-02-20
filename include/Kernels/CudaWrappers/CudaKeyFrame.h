#ifndef CUDA_KEYFRAME_H
#define CUDA_KEYFRAME_H

#include "CudaMapPoint.h"
#include "CudaKeyPoint.h"
#include "KeyFrame.h"
#include "../CudaUtils.h"

namespace MAPPING_DATA_WRAPPER {

class CudaKeyFrame {
    private:
        void initializeMemory();

    public:
        CudaKeyFrame();
        void setGPUAddress(CudaKeyFrame* ptr);
        void setMemory(ORB_SLAM3::KeyFrame* KF);
        void addMapPoint(ORB_SLAM3::MapPoint* mp, int idx);
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
        CudaKeyFrame* gpuAddr;

    private:
        std::vector<CudaMapPoint*> h_mvpMapPoints;
    };
}

#endif // CUDA_KEYFRAME_H