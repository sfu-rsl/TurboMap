#ifndef MAPPING_KERNEL_CONTROLLER_H
#define MAPPING_KERNEL_CONTROLLER_H

#include "KFCullingKernel.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaKeyFrameDrawer.h"
#include "CudaMapPointStorage.h"
#include "CudaUtils.h"
#include "FuseKernel.h"
#include "SearchForTriangulationKernel.h"
#include <memory> 

using namespace std;

class MappingKernelController{
public:
    static void setCUDADevice(int deviceID);
    
    static bool is_active;
    
    static void activate();

    static bool keyframeCullingOnGPU;
    static bool fuseKernelRunStatus;
    static bool searchForTriangulationOnGPU;

    static void setGPURunMode(bool keyframeCulling, bool FuseStatus, bool searchForTriangulation);

    static void initializeKernels();
    
    static void shutdownKernels();
    
    static void saveKernelsStats(const std::string &file_path);
    
    static void launchKeyframeCullingKernel(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int* h_kf_count, long unsigned int* h_indices,
                                                                            int* h_nMPs, int* h_nRedundantObservations);
    
    static void launchFuseKernel(ORB_SLAM3::KeyFrame &KF, const vector<ORB_SLAM3::MapPoint*> &vpMapPoints,
                            const float th, const bool bRight, int* h_bestDist, int* h_bestIdx,
                            ORB_SLAM3::GeometricCamera* pCamera, Sophus::SE3f Tcw, Eigen::Vector3f Ow);

    static void launchSearchForTriangulationKernel(
        ORB_SLAM3::KeyFrame* mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs, 
        bool mbMonocular, bool mbInertial, bool recentlyLost, bool mbIMU_BA2, 
        std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices, std::vector<size_t> &vpNeighKFsIndexes
    );

private:
    static bool memory_is_initialized;
    static MAPPING_DATA_WRAPPER::CudaKeyFrame *cudaKeyFramePtr;
    static std::unique_ptr<KFCullingKernel> mpKFCullingKernel;
    static std::unique_ptr<FuseKernel> mpFuseKernel;
    static std::unique_ptr<SearchForTriangulationKernel> mpSearchForTriangulationKernel;
};

#endif
