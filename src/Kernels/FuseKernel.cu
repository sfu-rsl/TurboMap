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
    if (!memory_is_initialized) 
        return;
    cudaFree(d_keyframe);
    cudaFreeHost(h_isEmpty);
    cudaFreeHost(h_mWorldPos);
    cudaFreeHost(h_mfMaxDistance);
    cudaFreeHost(h_mfMinDistance);
    cudaFreeHost(h_mNormalVector);
    cudaFreeHost(h_mDescriptor);
    cudaFree(d_isEmpty);
    cudaFree(d_mWorldPos);
    cudaFree(d_mfMaxDistance);
    cudaFree(d_mfMinDistance);
    cudaFree(d_mNormalVector);
}


__device__ void printDeviceData(bool* d_isEmpty, Eigen::Vector3f* d_mWorldPos, float* d_mfMaxDistance, float* d_mfMinDistance, Eigen::Vector3f* d_mNormalVector, uint8_t* d_mDescriptor, int idx) {
    printf("Device Data at index %d:\n", idx);

    // Print d_isEmpty
    printf("d_isEmpty: %d\n", d_isEmpty[idx]);

    // Print d_mWorldPos
    printf("d_mWorldPos: ", d_mWorldPos);

    printf("d_mfMaxDistance: %f\n", d_mfMaxDistance[idx]);
    printf("d_mfMinDistance: %f\n", d_mfMinDistance[idx]);

    // Print d_mNormalVector
    printf("d_mNormalVector: [%f, %f, %f]\n", d_mNormalVector);

    // Print d_mor
    printf("d_mDescriptor: ");
    for (int i = 0; i < DESCRIPTOR_SIZE; ++i) {
        printf("%d ", d_mDescriptor[idx * DESCRIPTOR_SIZE + i]);
    }
    printf("\n");
}



__global__ void fuseKernel(MAPPING_DATA_WRAPPER::CudaKeyFrame* d_keyframe,
                    bool *d_isEmpty,
                    Eigen::Vector3f *d_mWorldPos,
                    float *d_mfMaxDistance,
                    float *d_mfMinDistance,
                    Eigen::Vector3f *d_mNormalVector,
                    uint8_t *d_mDescriptor)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    printf("#####################################\n");

    printDeviceData(d_isEmpty, d_mWorldPos, d_mfMaxDistance, d_mfMinDistance, d_mNormalVector, d_mDescriptor, idx);


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

    std::vector<MAPPING_DATA_WRAPPER::CudaMapPoint> tmp_mvpMapPoints(mvpMapPoints_size);
    for (int i = 0; i < mvpMapPoints_size; ++i) {
            if (F.mvpMapPoints[i]) {
                CudaMapPoint cuda_mp(F.mvpMapPoints[i]);
                tmp_mvpMapPoints[i] = cuda_mp;
            } else {
                CudaMapPoint cuda_mp;
                tmp_mvpMapPoints[i] = cuda_mp;            
            }
    }
    checkCudaError(cudaMemcpy(mvpMapPoints, tmp_mvpMapPoints.data(), tmp_mvpMapPoints.size() * sizeof(CudaMapPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvpMapPoints to gpu");


    #pragma omp parallel for
    for (int i = 0; i < numPoints; ++i) {

        std::cout << i << std::endl;

        ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        std::cout << "here0\n";
        if((!pMP) || (pMP->IsInKeyFrame(&pKF)) || (pMP->isBad())){
            std::cout << "Inside if1\n";
            h_isEmpty[i] = true;
            std::cout << "Inside if2\n";
            continue;
        }
        std::cout << "here1\n";
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        std::cout << "here2\n";
        Eigen::Vector3f p3Dc = Tcw * p3Dw;
        std::cout << "here3\n";
        const float invz = 1/p3Dc(2);
        std::cout << "here4\n";
        const Eigen::Vector2f uv = pCamera->project(p3Dc);
        std::cout << "here5\n";

        if((p3Dc(2)<0.0f) || (!pKF.IsInImage(uv(0),uv(1))))
        {
            h_isEmpty[i] = true;
            continue;
        }

        h_isEmpty[i] = false;
        // h_mWorldPos = pMP->GetWorldPos();
        float mfMaxDistance = pMP->GetMaxDistanceInvariance();
        h_mfMaxDistance = &mfMaxDistance;

        float mfMinDistance = pMP->GetMinDistanceInvariance();
        h_mfMinDistance = &mfMinDistance;

        Eigen::Vector3f normalVector = pMP->GetNormal();
        h_mNormalVector = &normalVector;

        std::memcpy(&h_mDescriptor[i*DESCRIPTOR_SIZE], pMP->GetDescriptor().data, DESCRIPTOR_SIZE * sizeof(uint8_t));
    }


    std::cout << "Outside for\n";

    assert(h_isEmpty != nullptr);
    assert(h_mWorldPos != nullptr);
    assert(h_mfMaxDistance != nullptr);
    assert(h_mfMinDistance != nullptr);
    assert(h_mNormalVector != nullptr);
    assert(h_mDescriptor != nullptr);

    std::cout << "End of asserts\n";

    checkCudaError(cudaMemcpy(d_isEmpty, h_isEmpty, numPoints * sizeof(bool), cudaMemcpyHostToDevice), 
                "Failed to copy memory for d_isEmpty");
    std::cout << "me1\n";
    checkCudaError(cudaMemcpy(d_mWorldPos, h_mWorldPos, numPoints * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice), 
                "Failed to copy memory for d_mWorldPos");
    std::cout << "me2\n";
    checkCudaError(cudaMemcpy(d_mfMaxDistance, h_mfMaxDistance, numPoints * sizeof(float), cudaMemcpyHostToDevice), 
                "Failed to copy memory for d_mfMaxDistance");
    std::cout << "me3\n";
    checkCudaError(cudaMemcpy(d_mfMinDistance, h_mfMinDistance, numPoints * sizeof(float), cudaMemcpyHostToDevice), 
                "Failed to copy memory for d_mfMinDistance");
    std::cout << "me4\n";
    checkCudaError(cudaMemcpy(d_mNormalVector, h_mNormalVector, numPoints * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice), 
                "Failed to copy memory for d_mNormalVector");
    std::cout << "me5\n";
    checkCudaError(cudaMemcpy(d_mDescriptor, h_mDescriptor, numPoints * DESCRIPTOR_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice), 
                "Failed to copy memory for d_mDescriptor");

    std::cout << "End of memcpy\n";

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize -1) / blockSize;
    std::cout << "****************************************************\n";
    fuseKernel<<<numBlocks, blockSize>>>(d_keyframe,
                                        d_isEmpty,
                                        d_mWorldPos,
                                        d_mfMaxDistance,
                                        d_mfMinDistance,
                                        d_mNormalVector,
                                        d_mDescriptor);
    
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";

    checkCudaError(cudaDeviceSynchronize(), "[fuseKernel:] Kernel launch failed");  

}

void FuseKernel::saveStats(const string &file_path){

    std::cout << "Saving stats for FuseKernel" << std::endl;
}