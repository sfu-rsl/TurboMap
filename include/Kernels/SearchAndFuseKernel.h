#ifndef SEARCH_AND_FUSE_KERNEL_H
#define SEARCH_AND_FUSE_KERNEL_H

#include <iostream>
#include "CudaWrappers/CudaMapPoint.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaKeyFrameStorage.h"
#include "CudaUtils.h"
#include "CameraModels/GeometricCamera.h"
#include <Eigen/Core>
#include <csignal> 

#define MAX_CONNECTED_KF_COUNT 40


class SearchAndFuseKernel{

    public:
        void initialize();
        void shutdown();
        int launch(std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, float th,
                std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints, vector<ORB_SLAM3::MapPoint*> &vpReplacePoints);
        void origFuse(ORB_SLAM3::KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<ORB_SLAM3::MapPoint*> &vpPoints, const float th);
        int origDescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    private:
        bool memory_is_initialized;
        int *d_bestDists, *d_bestIdxs;
        int *bestDists, *bestIdxs;
        MAPPING_DATA_WRAPPER::CudaMapPoint *h_MapPoints, *d_MapPoints;
        MAPPING_DATA_WRAPPER::CudaKeyFrame **h_KeyFrames, **d_KeyFrames;
        Eigen::Vector3f *h_Ow, *d_Ow;
        Sophus::SE3f *h_Tcw, *d_Tcw;

};

#endif 