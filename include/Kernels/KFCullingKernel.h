#ifndef KEYFRAME_CULLING_KERNEL_H
#define KEYFRAME_CULLING_KERNEL_H

#include "KernelInterface.h"
#include "CudaUtils.h"
#include "../KeyFrame.h"
#include "CudaWrappers/CudaKeyframe.h"
#include <map>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_NUM_KEYFRAMES 400

class KFCullingKernel: public KernelInterface {
    
    public:
        KFCullingKernel(){memory_is_initialized=false;};
        void initialize() override;
        void shutdown() override;
        void saveStats(const std::string &file_path) override {};
        void launch() override { std::cout << "[KFCullingKernel:] provide input for kernel launch.\n"; };
        void launch(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int* h_kf_count, long unsigned int* h_indices,
                                                                            int* h_nMPs, int* h_nRedundantObservations);
    
    private:
        bool memory_is_initialized;
        MAPPING_DATA_WRAPPER::CudaKeyframe **d_keyframes, **h_keyframes;
        // MAPPING_DATA_WRAPPER::CudaMapPoint *d_mapPoints, *h_mapPoints;
        int *d_nMPs, *d_nRedundantObservations;
        int *empty_array[MAX_NUM_KEYFRAMES];
};

#endif