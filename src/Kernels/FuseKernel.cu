#include <iostream>
#include "Kernels/FuseKernel.h"


void FuseKernel::initialize() {
    if (memory_is_initialized) {
        return;
    }

    checkCudaError(cudaMalloc(&d_keyframe, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "Failed to allocate memory for d_keyframe");

    checkCudaError(cudaMallocHost(&h_isEmpty, MAX_NUM_MAPPOINTS * sizeof(bool)), "Failed to allocate memory for h_isEmpty");
    checkCudaError(cudaMallocHost(&h_mWorldPos, MAX_NUM_MAPPOINTS * sizeof(Eigen::Vector3f)), "Failed to allocate memory for h_mWorldPos");
    checkCudaError(cudaMallocHost(&h_mfMaxDistance, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for h_mfMaxDistance");
    checkCudaError(cudaMallocHost(&h_mfMinDistance, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for h_mfMinDistance");
    checkCudaError(cudaMallocHost(&h_mNormalVector, MAX_NUM_MAPPOINTS * sizeof(Eigen::Vector3f)), "Failed to allocate memory for h_mNormalVector");
    checkCudaError(cudaMallocHost(&h_mDescriptor, MAX_NUM_MAPPOINTS * DESCRIPTOR_SIZE * sizeof(uint8_t)), "Failed to allocate memory for h_mDescriptor");

    checkCudaError(cudaMalloc((void**)&d_isEmpty, MAX_NUM_MAPPOINTS * sizeof(bool)), "Failed to allocate memory for d_isEmpty");
    checkCudaError(cudaMalloc((void**)&d_mWorldPos, MAX_NUM_MAPPOINTS * sizeof(Eigen::Vector3f)), "Failed to allocate memory for d_mWorldPos");
    checkCudaError(cudaMalloc((void**)&d_mfMaxDistance, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for d_mfMaxDistance");
    checkCudaError(cudaMalloc((void**)&d_mfMinDistance, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for d_mfMinDistance");
    checkCudaError(cudaMalloc((void**)&d_mNormalVector, MAX_NUM_MAPPOINTS * sizeof(Eigen::Vector3f)), "Failed to allocate memory for d_mNormalVector");   
    checkCudaError(cudaMalloc((void**)&d_mDescriptor, MAX_NUM_MAPPOINTS * DESCRIPTOR_SIZE * sizeof(uint8_t)), "Failed to allocate memory for d_mDescriptor");   

    memory_is_initialized = true;
}

void FuseKernel::setKeyFrame(MAPPING_DATA_WRAPPER::CudaKeyFrame* cudaKeyFrame) {
    checkCudaError(cudaMemcpy(d_keyframe, cudaKeyFrame, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "Failed to copy KeyFrame to device");
}

void FuseKernel::shutdown() {
    if (memory_is_initialized) {
    }
}


__global__ void fuseKernel()
{

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

    #pragma omp parallel for
    for (int i = 0; i < numPoints; ++i) {
        ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        if((!pMP) || (pMP->IsInKeyFrame(pKF)) || (pMP->isBad())){
            h_isEmpty[i] = true;
            continue;
        }

        h_isEmpty[i] = false;
        h_mWorldPos = pMP->GetWorldPos();
        h_mfMaxDistance = pMP->GetMaxDistanceInvariance();
        h_mfMinDistance = pMP->GetMinDistanceInvariance();
        h_mNormalVector = pMP->GetNormal();
        std::memcpy(&h_mDescriptor[i*DESCRIPTOR_SIZE], pMP->GetDescriptor().data, DESCRIPTOR_SIZE * sizeof(uint8_t));
    }

    cudaMemcpy(d_isEmpty, h_isEmpty, numPoints * sizeof(bool), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mWorldPos, h_mWorldPos, numPoints * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mfMaxDistance, h_mfMaxDistance, numPoints * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mfMinDistance, h_mfMinDistance, numPoints * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mNormalVector, h_mNormalVector, numPoints * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mDescriptor, h_mDescriptor, numPoints * DESCRIPTOR_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice); 

}

void FuseKernel::saveStats(const string &file_path){

    std::cout << "Saving stats for FuseKernel" << std::endl;
}