#include <iostream>
#include "Kernels/FuseKernel.h"


void FuseKernel::initialize() {
    if (memory_is_initialized) {
        return;
    }

    checkCudaError(cudaMalloc(&d_keyframe, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame)), "Failed to allocate memory for d_keyframe");
    checkCudaError(cudaMallocHost(&h_uv, MAX_NUM_MAPPOINTS * sizeof(Eigen::Vector2f)), "Failed to allocate memory for h_uv");
    checkCudaError(cudaMallocHost(&h_ur, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for h_ur");

    checkCudaError(cudaMalloc((void**)&d_mvpMapPoints, MAX_NUM_MAPPOINTS * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint)), "Failed to allocate memory for d_mvpMapPoints");
    checkCudaError(cudaMalloc((void**)&d_uv, MAX_NUM_MAPPOINTS * sizeof(Eigen::Vector2f)), "Failed to allocate memory for d_uv");
    checkCudaError(cudaMalloc((void**)&d_ur, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for d_ur");

    checkCudaError(cudaMallocHost(&h_bestDist, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestDist");
    checkCudaError(cudaMallocHost(&h_bestIdx, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestIdx");      

    checkCudaError(cudaMalloc((void**)&d_bestDist, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestDist");
    checkCudaError(cudaMalloc((void**)&d_bestIdx, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestIdx");


    memory_is_initialized = true;
}

void FuseKernel::setKeyFrame(MAPPING_DATA_WRAPPER::CudaKeyFrame* CudaKeyFrame) {
    checkCudaError(cudaMemcpy(d_keyframe, CudaKeyFrame, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "Failed to copy KeyFrame to device");
}

void FuseKernel::shutdown() {
    if (!memory_is_initialized) 
        return;
    cudaFreeHost(h_uv);
    cudaFreeHost(h_ur);
    cudaFreeHost(h_bestDist);
    cudaFreeHost(h_bestIdx);

    cudaFree(d_uv);
    cudaFree(d_ur);
    cudaFree(d_bestDist);
    cudaFree(d_bestIdx);
    checkCudaError(cudaFree(d_keyframe), "Failed to free fuse kernel: d_keyframe");
    checkCudaError(cudaFree(d_mvpMapPoints),"Failed to free fuse kernel memory: d_mvpMapPoints");
}


__device__ int predictScale(float currentDist, float maxDistance, 
                        MAPPING_DATA_WRAPPER::CudaKeyFrame* pKF)
{
    float ratio = maxDistance/currentDist;

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}


__global__ void fuseKernel(MAPPING_DATA_WRAPPER::CudaKeyFrame* d_keyframe,
                        MAPPING_DATA_WRAPPER::CudaMapPoint* d_mvpMapPoints,
                        int numPoints, Eigen::Vector3f Ow, float th,
                        Eigen::Vector2f *d_uv, float *d_ur, bool bRight,
                        int* d_bestDist, int* d_bestIdx)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= numPoints)
        return;

    d_bestDist[idx] = 256;
    d_bestIdx[idx] = -1;

    const float maxDistance = d_mvpMapPoints[idx].mfMaxDistance;
    const float minDistance = d_mvpMapPoints[idx].mfMinDistance;
    Eigen::Vector3f p3Dw = d_mvpMapPoints[idx].mWorldPos;
    Eigen::Vector3f PO = p3Dw-Ow;
    const float dist3D = PO.norm();
    if(dist3D<minDistance || dist3D>maxDistance) {
        return;
    }
    Eigen::Vector3f Pn = d_mvpMapPoints[idx].mNormalVector;

    if(PO.dot(Pn)<0.5*dist3D)
    {
        return;
    }

    int nPredictedLevel = predictScale(dist3D, maxDistance, d_keyframe);
    const float radius = th * d_keyframe->mvScaleFactors[nPredictedLevel];

    const uint8_t* MPdescriptor = &d_mvpMapPoints[idx].mDescriptor[0];
    int bestDist=256;
    int bestIdx= -1;

    // ## GetFeaturesInArea Function ##
    float factorX = radius;
    float factorY = radius;
    float x = d_uv[idx].x();
    float y = d_uv[idx].y();

    const int nMinCellX = max(0,(int)floor((x - d_keyframe->mnMinX - factorX) * d_keyframe->mfGridElementWidthInv));
    if(nMinCellX >= d_keyframe->mnGridCols)
        return;
    
    const int nMaxCellX = min((int)d_keyframe->mnGridCols-1,(int)ceil((x - d_keyframe->mnMinX + factorX) * d_keyframe->mfGridElementWidthInv));
    if(nMaxCellX<0)
        return;
    
    const int nMinCellY = max(0,(int)floor((y - d_keyframe->mnMinY - factorY) * d_keyframe->mfGridElementHeightInv));
    if(nMinCellY >= d_keyframe->mnGridRows)
        return;
    
    const int nMaxCellY = min((int)d_keyframe->mnGridRows-1,(int)ceil((y - d_keyframe->mnMinY + factorY) * d_keyframe->mfGridElementHeightInv));
    if(nMaxCellY<0)
        return;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {   
            std::size_t* vCell;
            int vCell_size;
            if (!bRight) {
                vCell = &d_keyframe->flatMGrid[ix * d_keyframe->mnGridRows * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                vCell_size = d_keyframe->flatMGrid_size[ix * d_keyframe->mnGridRows + iy];
            } else {
                vCell = &d_keyframe->flatMGridRight[ix * d_keyframe->mnGridRows * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                vCell_size = d_keyframe->flatMGridRight_size[ix * d_keyframe->mnGridRows + iy];
            }

            for(size_t j=0, jend=vCell_size; j<jend; j++)
            {
                size_t temp_idx = vCell[j];

                const MAPPING_DATA_WRAPPER::CudaKeyPoint &kpUn = (d_keyframe->Nleft == -1) ? d_keyframe->mvKeysUn[temp_idx]
                                                                    : (!bRight) ? d_keyframe->mvKeys[temp_idx]
                                                                                : d_keyframe->mvKeysRight[temp_idx];
                
                const float distx = kpUn.ptx-x;
                const float disty = kpUn.pty-y;
                
                if(fabs(distx)<radius && fabs(disty)<radius){

                    const int &kpLevel= kpUn.octave;

                    if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                        continue;

                    if(d_keyframe->mvuRight[temp_idx]>=0)
                    {
                        // Check reprojection error in stereo
                        const float &kpx = kpUn.ptx;
                        const float &kpy = kpUn.pty;
                        const float &kpr = d_keyframe->mvuRight[temp_idx];
                        const float ex = x-kpx;
                        const float ey = y-kpy;
                        const float er = d_ur[idx]-kpr;
                        const float e2 = ex*ex+ey*ey+er*er;

                        if(e2 * d_keyframe->mvInvLevelSigma2[kpLevel]>7.8)
                            continue;
                    }
                    else
                    {
                        const float &kpx = kpUn.ptx;
                        const float &kpy = kpUn.pty;
                        const float ex = x-kpx;
                        const float ey = y-kpy;
                        const float e2 = ex*ex+ey*ey;
                        if(e2 * d_keyframe->mvInvLevelSigma2[kpLevel]>5.99)
                            continue;
                    }

                    if(bRight) temp_idx += d_keyframe->Nleft;

                    const uint8_t* dKF = &d_keyframe->mDescriptors[temp_idx * DESCRIPTOR_SIZE];

                    int dist = DescriptorDistance(MPdescriptor,dKF);

                    if(dist<bestDist)
                    {
                        bestDist = dist;
                        bestIdx = temp_idx;
                    }
                }
            }
        }
    }
    
    d_bestDist[idx] = bestDist;
    d_bestIdx[idx] = bestIdx;

}



void FuseKernel::launch(ORB_SLAM3::KeyFrame &pKF, const vector<ORB_SLAM3::MapPoint *> &vpMapPoints, const float th, const bool bRight, int* h_bestDist, int* h_bestIdx,
                        ORB_SLAM3::GeometricCamera* pCamera, Sophus::SE3f Tcw, Eigen::Vector3f Ow)
{

    // std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();


    if (!memory_is_initialized){
        initialize();
    }

    int numPoints = vpMapPoints.size();
    if(numPoints > MAX_NUM_MAPPOINTS) {
        cout << "[ERROR] FuseKernel::launchKernel: ] number of mappoints: " << numPoints << " is greater than MAX_NUM_MAPPOINTS: " << MAX_NUM_MAPPOINTS << "\n";
        raise(SIGSEGV);
    }

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

        h_uv[i] = uv;
        h_ur[i] = uv(0) - pKF.mbf * invz;
        MAPPING_DATA_WRAPPER::CudaMapPoint cuda_mp(pMP);
        h_mvpMapPoints[i] = cuda_mp;
    }

    cudaMemcpy(d_uv, h_uv, numPoints * sizeof(Eigen::Vector2f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ur, h_ur, numPoints * sizeof(float), cudaMemcpyHostToDevice); 
    checkCudaError(cudaMemcpy(d_mvpMapPoints, h_mvpMapPoints.data(), numPoints * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "FuseKernel:: Failed to copy mvpMapPoints to gpu");

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize -1) / blockSize;
    fuseKernel<<<numBlocks, blockSize>>>(d_keyframe, d_mvpMapPoints, numPoints, Ow, th, d_uv, d_ur, bRight, d_bestDist, d_bestIdx);
    
    checkCudaError(cudaDeviceSynchronize(), "[fuseKernel:] Kernel launch failed");  

    checkCudaError(cudaMemcpy(h_bestDist, d_bestDist, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDist back to host");
    checkCudaError(cudaMemcpy(h_bestIdx, d_bestIdx, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdx back to host");


    // std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();
    // double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();
    // std::cout << "total: " << total << std::endl;

}

void FuseKernel::saveStats(const string &file_path){

    std::cout << "Saving stats for FuseKernel" << std::endl;
}