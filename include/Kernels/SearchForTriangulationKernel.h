#ifndef SEARCHFORTRIANGULATIONKERNEL_H
#define SEARCHFORTRIANGULATIONKERNEL_H

#include "CudaUtils.h"
#include "CudaWrappers/CudaKeyPoint.h"
#include "CudaWrappers/CudaCamera.h"
#include "KernelInterface.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

#include "KeyFrame.h"
#include "Frame.h"
#include "MapPoint.h"
#include "GeometricCamera.h"
#include "KannalaBrandt8.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

#define MAX_NEIGHBOUR_COUNT 11
#define MAX_FEATURES_IN_WORD 100
#define MAX_FEAT_VEC_SIZE 100
#define MAP_POINT_COUNT_TH 50
#define MATCH_TH_LOW 50
#define CV_PI 3.1415926535897932384626433832795

class SearchForTriangulationKernel : public KernelInterface {
public:
    SearchForTriangulationKernel() { memory_is_initialized = false; };
    void initialize() override;
    void shutdown() override;
    void saveStats(const std::string &file_path) override;
    void launch() override { std::cout << "[SearchForTriangulationKernel:] provide input for kernel launch.\n"; };
    void launch(ORB_SLAM3::KeyFrame *mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs, 
                bool mbMonocular, bool mbInertial, bool recentlyLost, bool mbIMU_BA2, 
                std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices, std::vector<size_t> &vpNeighKFsIndexes);

private:
    void copyGPUKeypoints(TRACKING_DATA_WRAPPER::CudaKeyPoint* out, const std::vector<cv::KeyPoint> keypoints);
    void copyGPUCamera(MAPPING_DATA_WRAPPER::CudaCamera *out, ORB_SLAM3::GeometricCamera *camera);
    void copyFrameFeatVec(ORB_SLAM3::KeyFrame* kf, unsigned int* outFeatureVec, size_t* outFeatureVecIdxs, 
                          size_t* outFeatureVecLastIdx, size_t maxFeatVecSize);
    void convertToVectorOfPairs(int* gpuMatchedIdxs, int nn, int featVecSize, 
                                std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices);
    void printMatchedIndices(const std::vector<std::vector<std::pair<size_t, size_t>>> &allvMatchedIndices);
    void origCreateNewMapPoints(ORB_SLAM3::KeyFrame* mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs, 
                                bool mbMonocular, bool mbInertial, bool recentlyLost, bool mbIMU_BA2);
    void origSearchForTriangulation(ORB_SLAM3::KeyFrame* pKF1, ORB_SLAM3::KeyFrame* pKF2, 
                                    std::vector<std::pair<size_t,size_t>> &vMatchedPairs, 
                                    const bool bOnlyStereo, const bool bCoarse);
    int origDescriptorDistance(const cv::Mat &a, const cv::Mat &b);


private:
    std::vector<std::pair<long unsigned int, double>> data_wrap_time;
    std::vector<std::pair<long unsigned int, double>> input_data_wrap_time;
    std::vector<std::pair<long unsigned int, double>> input_data_transfer_time;
    std::vector<std::pair<long unsigned int, double>> kernel_exec_time;
    std::vector<std::pair<long unsigned int, double>> output_data_transfer_time;
    std::vector<std::pair<long unsigned int, double>> data_transfer_time;
    std::vector<std::pair<long unsigned int, double>> output_data_wrap_time;
    std::vector<std::pair<long unsigned int, double>> total_exec_time;

    long unsigned int frameCounter;

private:
    bool memory_is_initialized;
    unsigned int *d_currFrameFeatVec, *d_neighFramesfeatVec;
    size_t *d_currFrameFeatVecIdxs, *d_neighFramesfeatVecIdxs, *d_neighFramesFeatVecStartIdxs;
    size_t *d_currFrameFeatVecIdxCorrespondences, *d_neighFramesFeatVecIdxCorrespondences;
    int *d_neighFramesNLeft;
    Eigen::Matrix3f *d_Rll, *d_Rlr, *d_Rrl, *d_Rrr;
    Eigen::Vector3f *d_tll, *d_tlr, *d_trl, *d_trr;
    Eigen::Matrix3f *d_R12;
    Eigen::Vector3f *d_t12;
    bool *d_currFrameMapPointExists, *d_neighFramesMapPointExists;
    uchar *d_currFrameDescriptors, *d_neighFramesDescriptors;
    TRACKING_DATA_WRAPPER::CudaKeyPoint *d_currFrameMvKeysUn, *d_neighFramesMvKeysUn;
    TRACKING_DATA_WRAPPER::CudaKeyPoint *d_currFrameMvKeys, *d_neighFramesMvKeys;
    TRACKING_DATA_WRAPPER::CudaKeyPoint *d_currFrameMvKeysRight, *d_neighFramesMvKeysRight;
    float *d_currFrameMvuRight, *d_neighFramesMvuRight;
    MAPPING_DATA_WRAPPER::CudaCamera *d_neighFramesCamera1, *d_neighFramesCamera2;
    Eigen::Vector2f *d_ep;

    int *d_matchedPairIndexes;
};

#endif