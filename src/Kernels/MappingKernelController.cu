#include "Kernels/MappingKernelController.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [Mapping KernelController::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

bool MappingKernelController::is_active = false;
bool MappingKernelController::keyframeCullingOnGPU;
bool MappingKernelController::memory_is_initialized = false;
std::unique_ptr<KFCullingKernel> MappingKernelController::mpKFCullingKernel = std::make_unique<KFCullingKernel>();

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

void MappingKernelController::setGPURunMode(bool _keyframeCulling) {
    keyframeCullingOnGPU = _keyframeCulling;
}

void MappingKernelController::initializeKernels(){

    DEBUG_PRINT("Initializing Kernels");
    
    CudaKeyframeDrawer::initializeMemory();

    CudaMapPointStorage::initializeMemory();
    
    if(keyframeCullingOnGPU == 1)
        mpKFCullingKernel->initialize();

    checkCudaError(cudaDeviceSynchronize(), "[Mapping Kernel Controller:] Failed to initialize kernels.");
    memory_is_initialized = true;
}

void MappingKernelController::shutdownKernels(){

    DEBUG_PRINT("Shutting Kernels Down");

    if (memory_is_initialized) {
        CudaKeyframeDrawer::shutdown();
        CudaMapPointStorage::shutdown();
        if(keyframeCullingOnGPU == 1)
            mpKFCullingKernel->shutdown();
    }
}

void MappingKernelController::saveKernelsStats(const std::string &file_path){

    DEBUG_PRINT("Saving Kernels Stats");
    
    mpKFCullingKernel->saveStats(file_path);
}

void MappingKernelController::launchKeyframeCullingKernel(vector<ORB_SLAM3::KeyFrame*> vpLocalKeyFrames, int* h_kf_count, long unsigned int* h_indices,
                                                                            int* h_nMPs, int* h_nRedundantObservations) {

    DEBUG_PRINT("Launching Keyframe Culling Kernel"); 

    mpKFCullingKernel->launch(vpLocalKeyFrames,h_kf_count, h_indices, h_nMPs, h_nRedundantObservations);
}

// void MappingKernelController::addKeyframeToGPU(ORB_SLAM3::KeyFrame* KF) {
//     DEBUG_PRINT("Adding Keyframe To GPU Keyframe Storage."); 

//     CudaKeyframeDrawer::addCudaKeyframe(KF);
//     vector<ORB_SLAM3::MapPoint*> mvpMapPoints = KF->GetMapPointMatches();
//     int mvpMapPoints_size = mvpMapPoints.size();
//     for (int i = 0; i < mvpMapPoints_size; ++i) {
//         if (mvpMapPoints[i]) {
//             MAPPING_DATA_WRAPPER::CudaMapPoint* d_mp = CudaMapPointStorage::getCudaMapPoint(mvpMapPoints[i]->mnId);
//             if (d_mp == nullptr) {
//                 CudaMapPointStorage::addCudaMapPoint(mvpMapPoints[i]);
//             }
//         }
//     }
//     CudaKeyframeDrawer::setCudaKeyframeMapPoints(KF->mnId, mvpMapPoints);
// }

// void MappingKernelController::addMapPointToGPU(ORB_SLAM3::MapPoint* MP) {
//     DEBUG_PRINT("Adding Map Point To GPU Map Point Storage."); 

//     CudaMapPointStorage::addCudaMapPoint(MP);
// }
