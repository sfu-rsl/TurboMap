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
#include "CudaCamera.h"
#include "../CudaKeyFrameDrawer.h"

#define MAX_FEAT_VEC_SIZE 100
#define MAX_FEAT_PER_WORD 100

namespace MAPPING_DATA_WRAPPER {

#define KEYPOINTS_PER_CELL 20

class CudaKeyFrame {
    private:
        void initializeMemory();
        void copyGPUCamera(CudaCamera *out, ORB_SLAM3::GeometricCamera *camera);
        void copyFeatVec(unsigned int *out, int *outIndexes, DBoW2::FeatureVector inp);
    
    public:
        CudaKeyFrame();
        void setMemory(ORB_SLAM3::KeyFrame &KF);
        void setMemory(ORB_SLAM3::KeyFrame* KF);
        void setGPUAddress(CudaKeyFrame* ptr);
        void addMapPoint(ORB_SLAM3::MapPoint* mp, int idx);
        void addFeatureVector(DBoW2::FeatureVector featVec);
        void freeMemory();
    
    public:
        long unsigned int mnId;
        int Nleft;
        float mfLogScaleFactor;
        int mnScaleLevels;
        float mnMinX;
        float mnMinY;
        float mfGridElementWidthInv;
        float mfGridElementHeightInv;
        int mnGridCols;
        int mnGridRows;
        float mThDepth;
        float* mvDepth;

        size_t mvpMapPoints_size;
        CudaMapPoint** mvpMapPoints;

        CudaKeyFrame* gpuAddr;

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

        size_t flatMGrid_size[FRAME_GRID_COLS * FRAME_GRID_ROWS];
        std::size_t flatMGrid[FRAME_GRID_COLS * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL];        
        
        size_t flatMGridRight_size[FRAME_GRID_COLS * FRAME_GRID_ROWS];
        std::size_t flatMGridRight[FRAME_GRID_COLS * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL];

        CudaCamera camera1, camera2;

        int mFeatCount;
        unsigned int *mFeatVec;
        int *mFeatVecStartIndexes;
        
    private:
        std::vector<CudaMapPoint*> h_mvpMapPoints;
    
    };
}

#endif // CUDA_KEYFRAME_H