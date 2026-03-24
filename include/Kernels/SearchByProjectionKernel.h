#ifndef SEARCH_BY_PROJECTION_KERNEL_H
#define SEARCH_BY_PROJECTION_KERNEL_H

#include <iostream>
#include "CudaWrappers/CudaMapPoint.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "LoopClosingCudaKeyFrameStorage.h"
#include "CudaUtils.h"
#include "CameraModels/GeometricCamera.h"
#include <Eigen/Core>
#include <csignal> 



class SearchByProjectionKernel{

    public:
        void initialize();
        void shutdown();
        int launch(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs,
                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming);
        int launch2(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming);
        void mergedlaunch(ORB_SLAM3::KeyFrame* pKF, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, Sophus::Sim3<float> &Scw1,
                        const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs, std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming,
                        std::vector<ORB_SLAM3::MapPoint*> &vpMatched1, int th1, float ratioHamming1,
                        int &numProjMatches, int &numProjOptMatches);
        void merged3launch(vector<ORB_SLAM3::KeyFrame*> currentCovKFs, vector<Sophus::Sim3f> currentCovmScws, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                        int th, float ratioHamming, int* num_matches, int covKFsSize);
        void origSearchByProjection(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs,
                                       std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming);
        void origSearchByProjection2(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming);
        int origDescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    
    private:
        bool memory_is_initialized;
        int *d_bestDists, *d_bestIdxs;
        int *bestDists, *bestIdxs;
        LOOP_DATA_WRAPPER::CudaMapPoint *h_MapPoints, *d_MapPoints;
        LOOP_DATA_WRAPPER::CudaKeyFrame *h_KeyFrame, *d_KeyFrame;
        LOOP_DATA_WRAPPER::CudaKeyFrame **h_KeyFrames, **d_KeyFrames;
        Eigen::Vector3f *h_Ow, *d_Ow;
        Sophus::SE3f *h_Tcw, *d_Tcw;
        
};

#endif