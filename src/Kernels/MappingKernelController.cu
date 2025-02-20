#include "Kernels/MappingKernelController.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [Mapping KernelController::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

bool MappingKernelController::is_active = false;
bool MappingKernelController::keyframeCullingOnGPU;
bool MappingKernelController::fuseKernelRunStatus;
bool MappingKernelController::memory_is_initialized = false;
std::unique_ptr<KFCullingKernel> MappingKernelController::mpKFCullingKernel = std::make_unique<KFCullingKernel>();
std::unique_ptr<FuseKernel> MappingKernelController::mpFuseKernel = std::make_unique<FuseKernel>();
MAPPING_DATA_WRAPPER::CudaKeyFrame* MappingKernelController::cudaKeyFramePtr;


void MappingKernelController::setCUDADevice(int deviceID) {
    cudaSetDevice(deviceID);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    printf("Using device %d: %s\n", deviceID, deviceProp.name);
}

void MappingKernelController::activate() {
    is_active = true;
    MappingKernelController::setGPURunMode(1);
}

void MappingKernelController::setGPURunMode(bool _keyframeCulling, bool _fuseStatus) {
    keyframeCullingOnGPU = _keyframeCulling;
    fuseKernelRunStatus = _fuseStatus;
}

void MappingKernelController::initializeKernels(){

    DEBUG_PRINT("Initializing Kernels");
    
    CudaKeyFrameDrawer::initializeMemory();

    CudaMapPointStorage::initializeMemory();

    cudaKeyFramePtr = new MAPPING_DATA_WRAPPER::CudaKeyFrame();

    if(keyframeCullingOnGPU == 1)
        mpKFCullingKernel->initialize();
    
    if(fuseKernelRunStatus == 1)
        mpFuseKernel->initialize();

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
        if(keyframeCullingOnGPU == 1)
            mpKFCullingKernel->shutdown();
        if(fuseKernelRunStatus == 1)
            mpFuseKernel->shutdown();
    }
}

void MappingKernelController::saveKernelsStats(const std::string &file_path){

    DEBUG_PRINT("Saving Kernels Stats");
    
    mpKFCullingKernel->saveStats(file_path);
    mpFuseKernel->saveStats(file_path);
}

void MappingKernelController::launchKeyframeCullingKernel(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int* h_kf_count, long unsigned int* h_indices,
                                                                            int* h_nMPs, int* h_nRedundantObservations) {

    DEBUG_PRINT("Launching Keyframe Culling Kernel"); 

    mpKFCullingKernel->launch(vpLocalKeyFrames,h_kf_count, h_indices, h_nMPs, h_nRedundantObservations);
}

void MappingKernelController::launchFuseKernel(ORB_SLAM3::KeyFrame &KF, const vector<ORB_SLAM3::MapPoint*> &vpMapPoints,
                                                    const float th, const bool bRight, int* h_bestDist, int* h_bestIdx,
                                                    ORB_SLAM3::GeometricCamera* pCamera, Sophus::SE3f Tcw, Eigen::Vector3f Ow) {

    cudaKeyFramePtr->setMemory(KF);
    mpFuseKernel->setKeyFrame(cudaKeyFramePtr);
    
    mpFuseKernel->launch(KF, vpMapPoints, th, bRight, h_bestDist, h_bestIdx, pCamera, Tcw, Ow);
}