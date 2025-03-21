#ifndef FUSE_KERNEL_H
#define FUSE_KERNEL_H

#include "KernelInterface.h"
#include <iostream>
#include "CudaWrappers/CudaMapPoint.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaKeyFrameDrawer.h"
#include "CudaUtils.h"
#include "CameraModels/GeometricCamera.h"
#include <Eigen/Core>
#include <csignal> 

class FuseKernel: public KernelInterface {

    public:
        FuseKernel() { memory_is_initialized = false; };
        void initialize() override;
        void shutdown() override;
        void launch() override { std::cout << "[FuseKernel:] provide input for kernel launch.\n"; };
        void launch(ORB_SLAM3::KeyFrame *neighKF, ORB_SLAM3::KeyFrame *currKF, const float th, const bool bRight,
                    int* h_bestDist, int* h_bestIdx, ORB_SLAM3::GeometricCamera* pCamera, Sophus::SE3f Tcw, Eigen::Vector3f Ow);
        void origFuse(ORB_SLAM3::KeyFrame *pKF, const vector<ORB_SLAM3::MapPoint*> &vpMapPoints, const float th, const bool bRight);
        int origDescriptorDistance(const cv::Mat &a, const cv::Mat &b);
        void saveStats(const string &file_path) override;

    private:
        bool memory_is_initialized;
        int *h_bestDist, *h_bestIdx;
        int *d_bestDist, *d_bestIdx;
};

#endif 