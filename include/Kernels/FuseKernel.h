#ifndef FUSE_KERNEL_H
#define FUSE_KERNEL_H

#include "KernelInterface.h"
#include <iostream>
#include "CudaWrappers/CudaMapPoint.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaUtils.h"
#include <Eigen/Core>


#define MAX_NUM_MAPPOINTS 25000
#define MAX_NUM_KEYPOINTS 25



class FuseKernel: public KernelInterface {

    public:
        FuseKernel() {memory_is_initialized=false; };
        void initialize() override;
        void shutdown() override;
        void launch() override { std::cout << "[FuseKernel:] provide input for kernel launch.\n"; };
        void launch(ORB_SLAM3::KeyFrame &pKF, const vector<ORB_SLAM3::MapPoint *> &vpMapPoints, const float th, const bool bRight);
        void setKeyFrame(MAPPING_DATA_WRAPPER::CudaKeyFrame* cudaKeyFrame);
        void saveStats(const string &file_path) override;

    private:
        bool memory_is_initialized;
        MAPPING_DATA_WRAPPER::CudaKeyFrame *d_keyframe, *h_keyframe;
        bool *h_isEmpty, *d_isEmpty;
        Eigen::Vector3f *h_mWorldPos, *d_mWorldPos;
        float *h_mfMaxDistance, *d_mfMaxDistance;
        float *h_mfMinDistance, *d_mfMinDistance;
        Eigen::Vector3f *h_mNormalVector, *d_mNormalVector;
        uint8_t *h_mDescriptor, *d_mDescriptor;


    public:

};


#endif 