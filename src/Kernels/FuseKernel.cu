#include <iostream>
#include "Kernels/FuseKernel.h"


void FuseKernel::initialize() {
    if (memory_is_initialized) {
        return;
    }

    checkCudaError(cudaMalloc(&d_keyframe, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "Failed to allocate memory for d_keyframe");

    checkCudaError(cudaMalloc((void**)&d_mvpMapPoints, MAX_NUM_MAPPOINTS * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint)), "Failed to allocate memory for d_mvpMapPoints");

    memory_is_initialized = true;
}

void FuseKernel::setKeyFrame(MAPPING_DATA_WRAPPER::CudaKeyFrame* cudaKeyFrame) {
    checkCudaError(cudaMemcpy(d_keyframe, cudaKeyFrame, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "Failed to copy KeyFrame to device");
}

void FuseKernel::shutdown() {
    if (!memory_is_initialized) 
        return;
    checkCudaError(cudaFree(d_keyframe), "Failed to free fuse kernel: d_keyframe");
    checkCudaError(cudaFree(d_mvpMapPoints),"Failed to free fuse kernel memory: d_mvpMapPoints");
}


__device__ void printDeviceData(
    const MAPPING_DATA_WRAPPER::CudaKeyFrame* d_keyframe,
    const MAPPING_DATA_WRAPPER::CudaMapPoint* d_mvpMapPoints,
    unsigned int idx
) {
    Check if the index is within bounds for the map points array
    if (idx == 0) {
        printf("----- KeyFrame Data -----\n");
        printf("mnId: %lu\n", d_keyframe->mnId);
        printf("fx: %f, fy: %f, cx: %f, cy: %f\n", d_keyframe->fx, d_keyframe->fy, d_keyframe->cx, d_keyframe->cy);
        printf("mbf: %f, Nleft: %d\n", d_keyframe->mbf, d_keyframe->Nleft);

        // Print mvScaleFactors
        printf("mvScaleFactors_size: %lu\n", d_keyframe->mvScaleFactors_size);
        for (size_t i = 0; i < d_keyframe->mvScaleFactors_size; ++i) {
            printf("mvScaleFactors[%lu]: %f\n", i, d_keyframe->mvScaleFactors[i]);
        }

        // Print mDescriptors (as an example, print the first few bytes)
        if (d_keyframe->mDescriptors) {
            printf("mDescriptors (first 10 bytes): ");
            for (int i = 0; i < min(d_keyframe->mDescriptor_rows, 10); ++i) {
                printf("%u ", d_keyframe->mDescriptors[i]);
            }
            printf("\n");
        }
    }

    // Check if idx is within bounds of d_mvpMapPoints array
    printf("----- MapPoint Data (idx: %u) -----\n", idx);
    if (!d_mvpMapPoints[idx].isEmpty) {
        printf("mnId: %lu\n", d_mvpMapPoints[idx].mnId);
        printf("mWorldPos: [%f, %f, %f]\n",
               d_mvpMapPoints[idx].mWorldPos.x(),
               d_mvpMapPoints[idx].mWorldPos.y(),
               d_mvpMapPoints[idx].mWorldPos.z());
        printf("mfMaxDistance: %f, mfMinDistance: %f\n",
               d_mvpMapPoints[idx].mfMaxDistance,
               d_mvpMapPoints[idx].mfMinDistance);
        printf("mNormalVector: [%f, %f, %f]\n",
               d_mvpMapPoints[idx].mNormalVector.x(),
               d_mvpMapPoints[idx].mNormalVector.y(),
               d_mvpMapPoints[idx].mNormalVector.z());
        printf("mDescriptor (first 10 bytes): ");
        for (int i = 0; i < 10; ++i) {
            printf("%u ", d_mvpMapPoints[idx].mDescriptor[i]);
        }
        printf("\n");
    }
    else {
        printf("MapPoint at idx %u is empty.\n", idx);
    }
}




__global__ void fuseKernel(MAPPING_DATA_WRAPPER::CudaKeyFrame* d_keyframe,
                        MAPPING_DATA_WRAPPER::CudaMapPoint* d_mvpMapPoints)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // printDeviceData(d_keyframe, d_mvpMapPoints, idx);


}



void FuseKernel::launch(ORB_SLAM3::KeyFrame &pKF, const vector<ORB_SLAM3::MapPoint *> &vpMapPoints, const float th, const bool bRight)
{
    if (!memory_is_initialized){
        initialize();
    }

    int numPoints = vpMapPoints.size();
    if(numPoints > MAX_NUM_MAPPOINTS) {
        cout << "[ERROR] FuseKernel::launchKernel: ] number of mappoints: " << numPoints << " is greater than MAX_NUM_MAPPOINTS: " << MAX_NUM_MAPPOINTS << "\n";
        raise(SIGSEGV);
    }

    ORB_SLAM3::GeometricCamera* pCamera;
    Sophus::SE3f Tcw;
    Eigen::Vector3f Ow;

    if(bRight){
        Tcw = pKF.GetRightPose();
        Ow = pKF.GetRightCameraCenter();
        pCamera = pKF.mpCamera2;
    }
    else{
        Tcw = pKF.GetPose();
        Ow = pKF.GetCameraCenter();
        pCamera = pKF.mpCamera;
    }

    std::cout << "Number of Mappoints is: "<< numPoints << ".\n";

    std::vector<MAPPING_DATA_WRAPPER::CudaMapPoint> h_mvpMapPoints(numPoints);
    for (int i = 0; i < numPoints; ++i) {

        ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        if((!pMP) || (pMP->IsInKeyFrame(&pKF)) || (pMP->isBad())){
            continue;
        }
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc = Tcw * p3Dw;
        const float invz = 1/p3Dc(2);
        const Eigen::Vector2f uv = pCamera->project(p3Dc);

        if((p3Dc(2)<0.0f) || (!pKF.IsInImage(uv(0),uv(1))))
        {
            continue;
        }

        MAPPING_DATA_WRAPPER::CudaMapPoint cuda_mp(pMP);
        h_mvpMapPoints[i] = cuda_mp;
    }
    checkCudaError(cudaMemcpy(d_mvpMapPoints, h_mvpMapPoints.data(), numPoints * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "FuseKernel:: Failed to copy mvpMapPoints to gpu");

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize -1) / blockSize;
    std::cout << "****************************************************\n";
    fuseKernel<<<numBlocks, blockSize>>>(d_keyframe, d_mvpMapPoints);
    
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";

    checkCudaError(cudaDeviceSynchronize(), "[fuseKernel:] Kernel launch failed");  

}

void FuseKernel::saveStats(const string &file_path){

    std::cout << "Saving stats for FuseKernel" << std::endl;
}