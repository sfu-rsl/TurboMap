#ifndef MAPPING_KERNEL_CONTROLLER_H
#define MAPPING_KERNEL_CONTROLLER_H

#include "KFCullingKernel.h"
#include "CudaKeyframeDrawer.h"
#include "CudaMapPointStorage.h"
#include "CudaUtils.h"
#include <memory> 

using namespace std;

class MappingKernelController{
public:
    static void setCUDADevice(int deviceID);
    
    static bool is_active;
    
    static void activate();

    static bool keyframeCullingOnGPU;

    static void setGPURunMode(bool keyframeCulling);

    static void initializeKernels();
    
    static void shutdownKernels();
    
    static void saveKernelsStats(const std::string &file_path);
    
    static void launchKeyframeCullingKernel(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int* h_kf_count, long unsigned int* h_indices,
                                                                            int* h_nMPs, int* h_nRedundantObservations);
    
    // static void addMapPointToGPUStorage(ORB_SLAM3::MapPoint* MP);

    // static void addKeyframeToGPUStorage(ORB_SLAM3::KeyFrame* KF);

private:
    static bool memory_is_initialized;
    static std::unique_ptr<KFCullingKernel> mpKFCullingKernel;
};

#endif
