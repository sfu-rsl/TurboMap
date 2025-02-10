#ifndef FUSE_KERNEL_H
#define FUSE_KERNEL_H

#include "KernelInterface.h"
#include <iostream>
#include "CudaWrappers/CudaMapPoint.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaUtils.h"
#include "CameraModels/GeometricCamera.h"
#include <Eigen/Core>
#include <csignal> 


#define MAX_NUM_MAPPOINTS 25000
#define MAX_NUM_KEYPOINTS 25
#define KEYPOINTS_PER_CELL 20



class FuseKernel: public KernelInterface {

    public:
        FuseKernel() {memory_is_initialized=false; };
        void initialize() override;
        void shutdown() override;
        void launch() override { std::cout << "[FuseKernel:] provide input for kernel launch.\n"; };
        void launch(ORB_SLAM3::KeyFrame &pKF, const vector<ORB_SLAM3::MapPoint *> &vpMapPoints, const float th, const bool bRight,
                    int* h_bestDist, int* h_bestIdx, ORB_SLAM3::GeometricCamera* pCamera,
                    Sophus::SE3f Tcw, Eigen::Vector3f Ow);
        void setKeyFrame(MAPPING_DATA_WRAPPER::CudaKeyFrame* CudaKeyFrame);
        void saveStats(const string &file_path) override;

    private:
        bool memory_is_initialized;
        Eigen::Vector2f *h_uv, *d_uv;
        float *h_ur, *d_ur;
        MAPPING_DATA_WRAPPER::CudaKeyFrame *d_keyframe, *h_keyframe;
        MAPPING_DATA_WRAPPER::CudaMapPoint *d_mvpMapPoints, *h_mvpMapPoints;
        int *h_bestDist, *h_bestIdx;
        int *d_bestDist, *d_bestIdx;

    public:

};


#endif 