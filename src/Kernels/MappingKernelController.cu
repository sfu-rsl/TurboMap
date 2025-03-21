#include "Kernels/MappingKernelController.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [Mapping KernelController::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

bool MappingKernelController::is_active = false;
bool MappingKernelController::keyframeCullingOnGPU;
bool MappingKernelController::fuseOnGPU;
bool MappingKernelController::searchForTriangulationOnGPU;
bool MappingKernelController::memory_is_initialized = false;
std::unique_ptr<KFCullingKernel> MappingKernelController::mpKFCullingKernel = std::make_unique<KFCullingKernel>();
std::unique_ptr<FuseKernel> MappingKernelController::mpFuseKernel = std::make_unique<FuseKernel>();
std::unique_ptr<SearchForTriangulationKernel> MappingKernelController::mpSearchForTriangulationKernel = std::make_unique<SearchForTriangulationKernel>();
MAPPING_DATA_WRAPPER::CudaKeyFrame* MappingKernelController::cudaKeyFramePtr;


void MappingKernelController::setCUDADevice(int deviceID) {
    cudaSetDevice(deviceID);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    printf("Using device %d: %s\n", deviceID, deviceProp.name);
}

void MappingKernelController::activate() {
    is_active = true;
    // MappingKernelController::setGPURunMode(0, 1, 0);
}

void MappingKernelController::setGPURunMode(bool _keyframeCulling, bool _fuseStatus, bool _searchForTriangulation) {
    keyframeCullingOnGPU = _keyframeCulling;
    fuseOnGPU = _fuseStatus;
    searchForTriangulationOnGPU = _searchForTriangulation;
}

void MappingKernelController::initializeKernels(){

    DEBUG_PRINT("Initializing Kernels");
    
    CudaKeyFrameDrawer::initializeMemory();

    CudaMapPointStorage::initializeMemory();

    cudaKeyFramePtr = new MAPPING_DATA_WRAPPER::CudaKeyFrame();

    if (keyframeCullingOnGPU == 1)
        mpKFCullingKernel->initialize();
    
    if (fuseOnGPU == 1)
        mpFuseKernel->initialize();

    if (searchForTriangulationOnGPU)
        mpSearchForTriangulationKernel->initialize();

    checkCudaError(cudaDeviceSynchronize(), "[Mapping Kernel Controller:] Failed to initialize kernels.");
    memory_is_initialized = true;
}

void MappingKernelController::shutdownKernels(){

    DEBUG_PRINT("Shutting Kernels Down");

    if (memory_is_initialized) {
        CudaKeyFrameDrawer::shutdown();
        CudaMapPointStorage::shutdown();
        cudaKeyFramePtr->freeMemory();
        delete cudaKeyFramePtr;
        if (keyframeCullingOnGPU == 1)
            mpKFCullingKernel->shutdown();
        if (fuseOnGPU == 1)
            mpFuseKernel->shutdown();
        if (searchForTriangulationOnGPU == 1)
            mpSearchForTriangulationKernel->shutdown();
    }
}

void MappingKernelController::saveKernelsStats(const std::string &file_path){

    DEBUG_PRINT("Saving Kernels Stats");
    
    mpKFCullingKernel->saveStats(file_path);
    mpFuseKernel->saveStats(file_path);
    mpSearchForTriangulationKernel->saveStats(file_path);
}

void MappingKernelController::launchKeyframeCullingKernel(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int* h_nMPs, int* h_nRedundantObservations) {

    DEBUG_PRINT("Launching Keyframe Culling Kernel"); 

    mpKFCullingKernel->launch(vpLocalKeyFrames, h_nMPs, h_nRedundantObservations);
}

void MappingKernelController::launchFuseKernel(ORB_SLAM3::KeyFrame *neighKF, ORB_SLAM3::KeyFrame *currKF,
                                               const float th, const bool bRight, int* h_bestDist, int* h_bestIdx,
                                               ORB_SLAM3::GeometricCamera* pCamera, Sophus::SE3f Tcw, Eigen::Vector3f Ow) {

    DEBUG_PRINT("Launching Fuse Kernel");
    
    mpFuseKernel->launch(neighKF, currKF, th, bRight, h_bestDist, h_bestIdx, pCamera, Tcw, Ow);
}

void MappingKernelController::launchSearchForTriangulationKernel(
    ORB_SLAM3::KeyFrame* mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs, 
    bool mbMonocular, bool mbInertial, bool recentlyLost, bool mbIMU_BA2, 
    std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices, 
    std::vector<size_t> &vpNeighKFsIndexes
) {
    mpSearchForTriangulationKernel->launch(mpCurrentKeyFrame, vpNeighKFs, mbMonocular, mbInertial, recentlyLost, mbIMU_BA2, allvMatchedIndices, vpNeighKFsIndexes);
}
