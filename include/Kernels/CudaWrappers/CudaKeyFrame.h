#ifndef CUDA_KEYFRAME_H
#define CUDA_KEYFRAME_H

#include <vector>
#include "Frame.h"
#include "../CudaUtils.h"
#include "CudaMapPoint.h"
#include "CudaKeyPoint.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include "../StereoMatchKernel.h"

namespace MAPPING_DATA_WRAPPER {

// CudaKeyFrameMemorySpace
class CudaKeyFrame {
    private:
        void initializeMemory();

    private:
        bool mvKeysIsOnGpu, mvKeysRightIsOnGpu, mDescriptorsIsOnGpu;
    
    public:
        CudaKeyFrame();
        void setMemory(const ORB_SLAM3::KeyFrame &KF);
        void setMvKeys(CudaKeyPoint* const &_mvKeys);
        void setMvKeysRight(CudaKeyPoint* const &_mvKeysRight);
        void setMDescriptors(uint8_t* const &_mDescriptors);
        const uint8_t* getMDescriptors() const { return mDescriptors; };
        void freeMemory();
    
    public:
        long unsigned int mnId;
        float fx;
        float fy;
        float cx;
        float cy;
        float mbf;
        int Nleft;

        size_t mvScaleFactors_size;
        float* mvScaleFactors;

        size_t mvKeys_size, mvKeysRight_size, mvKeysUn_size;
        const CudaKeyPoint *mvKeys, *mvKeysRight;
        CudaKeyPoint *mvKeysUn;

        size_t mvuRight_size;
        float* mvuRight;

        size_t mvInvLevelSigma2_size;
        float* mvInvLevelSigma2;

        int mDescriptor_rows;
        const uint8_t* mDescriptors;
        
        // float mpCamera_mvParameters[8];
        
    };
}

#endif // CUDA_KEYFRAME_H