#include <iostream>
#include "Kernels/SearchAndFuseKernel.h"
#include "Kernels/LoopClosingKernelController.h"

void SearchAndFuseKernel::initialize()
{
    if (memory_is_initialized)
        return;

    size_t mapPointVecSize = 3000;
    size_t connectedKFSize = 100;

    cudaMallocHost((void**)&h_MapPoints, mapPointVecSize * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint));
    cudaMallocHost((void**)&h_KeyFrames, connectedKFSize * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame));
    cudaMallocHost((void**)&h_Ow, connectedKFSize * sizeof(Eigen::Vector3f));
    cudaMallocHost((void**)&h_Tcw, connectedKFSize * sizeof(Sophus::SE3f));
    cudaMallocHost((void**)&bestDists, connectedKFSize * mapPointVecSize * sizeof(int));
    cudaMallocHost((void**)&bestIdxs, connectedKFSize * mapPointVecSize * sizeof(int));

    cudaMalloc(&d_MapPoints, mapPointVecSize * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint));
    cudaMalloc(&d_KeyFrames, connectedKFSize * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame));
    cudaMalloc(&d_Ow, connectedKFSize * sizeof(Eigen::Vector3f));
    cudaMalloc(&d_Tcw, connectedKFSize * sizeof(Sophus::SE3f));
    cudaMalloc(&d_bestDists, connectedKFSize * mapPointVecSize * sizeof(int));
    cudaMalloc(&d_bestIdxs, connectedKFSize * mapPointVecSize * sizeof(int));

    memory_is_initialized = true;
}

void SearchAndFuseKernel::shutdown()
{
    if (!memory_is_initialized) 
        return;

    cudaFreeHost(h_MapPoints);
    cudaFreeHost(h_KeyFrames);
    cudaFreeHost(h_Ow);
    cudaFreeHost(h_Tcw);
    cudaFreeHost(bestDists);
    cudaFreeHost(bestIdxs);
    cudaFree(d_MapPoints);
    cudaFree(d_KeyFrames);
    cudaFree(d_Ow);
    cudaFree(d_Tcw);
    cudaFree(d_bestDists);
    cudaFree(d_bestIdxs);

    memory_is_initialized = false;
}

__device__ inline Eigen::Vector2f KannalaBrandt8Project(const Eigen::Vector3f &v3D, float* mvParameters)
{
    const float x2_plus_y2 = v3D[0] * v3D[0] + v3D[1] * v3D[1];
    const float theta = atan2f(sqrtf(x2_plus_y2), v3D[2]);
    const float psi = atan2f(v3D[1], v3D[0]);

    const float theta2 = theta * theta;
    const float theta3 = theta * theta2;
    const float theta5 = theta3 * theta2;
    const float theta7 = theta5 * theta2;
    const float theta9 = theta7 * theta2;
    const float r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5
                         + mvParameters[6] * theta7 + mvParameters[7] * theta9;

    Eigen::Vector2f res;
    res[0] = mvParameters[0] * r * cos(psi) + mvParameters[2];
    res[1] = mvParameters[1] * r * sin(psi) + mvParameters[3];
    return res;

    // Eigen::Vector2f res;
    // res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
    // res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];
    // return res;
}


__device__ inline bool isInImage1(MAPPING_DATA_WRAPPER::CudaKeyFrame* keyframe, const float &x, const float &y)
{
    return (x>=keyframe->mnMinX && x<keyframe->mnMaxX && y>=keyframe->mnMinY && y<keyframe->mnMaxY);
}

__device__ int predictScale1(float currentDist, float maxDistance, MAPPING_DATA_WRAPPER::CudaKeyFrame* pKF)
{
    float ratio = maxDistance/currentDist;
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

__global__ void searchAndFuseKernel(Eigen::Vector3f* Ow, Sophus::SE3f *Tcw, 
                            MAPPING_DATA_WRAPPER::CudaKeyFrame** connectedKFs, MAPPING_DATA_WRAPPER::CudaMapPoint* mapPoints,
                            int chunkNumKFs, int totalNumKFs, int baseKFIdx, int numPoints, float th,
                            int* bestDists, int* bestIdxs) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIdx = numPoints * totalNumKFs;
    int connectedKFIdx = idx / numPoints;
    int mapPointIdx = idx % numPoints;
    
    if (idx >= maxIdx || connectedKFIdx >= totalNumKFs || mapPointIdx >= numPoints){
        return;
    }

    bestDists[idx] = 256;
    bestIdxs[idx] = -1;

    MAPPING_DATA_WRAPPER::CudaMapPoint& pMP = mapPoints[mapPointIdx];
    MAPPING_DATA_WRAPPER::CudaKeyFrame *connectedKF = connectedKFs[connectedKFIdx];

    // printf("idx: %llu, pMP.mnId: %llu, keyframe.mnId: %llu, connectedKF->mapPointsId_size: %d\n", idx, pMP.mnId, connectedKF->mnId, connectedKF->mapPointsId_size);

    Sophus::SE3f currTcw = Tcw[connectedKFIdx];
    Eigen::Vector3f currOw = Ow[connectedKFIdx];

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = currTcw * p3Dw;

    if (p3Dc(2) < 0.0f){
        return;
    }

    Eigen::Vector2f uv;

    uv = KannalaBrandt8Project(p3Dc, connectedKF->camera1.mvParameters);
    
    if ((!isInImage1(connectedKF, uv(0), uv(1)))){
        return;
    }

    const float maxDistance = 1.2f * pMP.mfMaxDistance;
    const float minDistance = 0.8f * pMP.mfMinDistance;
    Eigen::Vector3f PO = p3Dw - currOw;
    const float dist3D = PO.norm();

    if (dist3D < minDistance || dist3D > maxDistance)
        return;
    
    Eigen::Vector3f Pn = pMP.mNormalVector;
    if (PO.dot(Pn) < 0.5*dist3D)
        return;

    int nPredictedLevel = predictScale1(dist3D, pMP.mfMaxDistance, connectedKF);
    const float radius = th * connectedKF->mvScaleFactors[nPredictedLevel];

    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];
    int bestDist = 256;
    int bestIdx = -1;

    float factorX = radius;
    float factorY = radius;
    float x = uv(0);
    float y = uv(1);

    const int nMinCellX = max(0,(int)floor((x - connectedKF->mnMinX - factorX) * connectedKF->mfGridElementWidthInv));
    if (nMinCellX >= connectedKF->mnGridCols) 
        return;
    
    const int nMaxCellX = min((int)connectedKF->mnGridCols-1,(int)ceil((x - connectedKF->mnMinX + factorX) * connectedKF->mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return;
    
    const int nMinCellY = max(0,(int)floor((y - connectedKF->mnMinY - factorY) * connectedKF->mfGridElementHeightInv));
    if (nMinCellY >= connectedKF->mnGridRows)
        return;
    
    const int nMaxCellY = min((int)connectedKF->mnGridRows-1,(int)ceil((y - connectedKF->mnMinY + factorY) * connectedKF->mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return;
    
    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {   
            std::size_t* vCell;
            int vCell_size;
            
            vCell = &connectedKF->flatMGrid[ix * connectedKF->mnGridRows * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
            vCell_size = connectedKF->flatMGrid_size[ix * connectedKF->mnGridRows + iy];
            
            for (size_t j=0, jend=vCell_size; j<jend; j++) {
                size_t temp_idx = vCell[j];

                const MAPPING_DATA_WRAPPER::CudaKeyPoint &kpUn = (connectedKF->Nleft == -1) ? connectedKF->mvKeysUn[temp_idx]
                                                                                        : (!false) ? connectedKF->mvKeys[temp_idx]
                                                                                                    : connectedKF->mvKeysRight[temp_idx];
                
                const float distx = kpUn.ptx-x;
                const float disty = kpUn.pty-y;

                if (fabs(distx) < radius && fabs(disty) < radius) {
                    const int &kpLevel= connectedKF->mvKeysUn[temp_idx].octave;

                    if (kpLevel < nPredictedLevel-1 || kpLevel > nPredictedLevel)
                        continue;
                    
                    const uint8_t* dKF = &connectedKF->mDescriptors[temp_idx * DESCRIPTOR_SIZE];

                    int dist = DescriptorDistance(MPdescriptor,dKF);

                    if (dist<bestDist) {
                        bestDist = dist;
                        bestIdx = temp_idx;
                    }
                } 
            }
        }
    }
    
    bestDists[idx] = bestDist;
    bestIdxs[idx] = bestIdx;
}


int SearchAndFuseKernel::launch(std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, float th,
                        std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints, vector<ORB_SLAM3::MapPoint*> &vpReplacePoints)
{
    std::ofstream timing("./test/timing.txt", std::ios::app);
    // auto start1 = std::chrono::high_resolution_clock::now();
    if (!memory_is_initialized)
        initialize();
    // auto end1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
    // timing << "? Initialization: " << elapsed1.count() << " ms" << std::endl;

    const int TH_LOW = 50;
    int numValidPoints = 0;
    int connectedKFSize = connectedKFs.size();
    size_t mapPointVecSize = vpMapPoints.size();
    timing << "connectedKFSize: " << connectedKFSize << std::endl;

    // auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < connectedKFSize; i++) {
        h_Tcw[i] = Sophus::SE3f(connectedScws[i].rotationMatrix(),connectedScws[i].translation()/connectedScws[i].scale());
        h_Ow[i] = h_Tcw[i].inverse().translation();
    }
    // auto end2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
    // timing << "? h_Tcw: " << elapsed2.count() << " ms" << std::endl;

    // auto start3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < mapPointVecSize; i++) {
        ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;
        else {
            h_MapPoints[numValidPoints] = MAPPING_DATA_WRAPPER::CudaMapPoint(pMP);
            numValidPoints++;
        }
    }
    // auto end3 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
    // timing << "? h_MapPoints: " << elapsed3.count() << " ms" << std::endl;

    // auto start4 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < connectedKFSize; i++){
        ORB_SLAM3::KeyFrame* pKF = connectedKFs[i];
        h_KeyFrames[i] = CudaKeyFrameStorage::getCudaKeyFrame(pKF->mnId);
        if (h_KeyFrames[i] == nullptr){
            h_KeyFrames[i] = CudaKeyFrameStorage::addCudaKeyFrame(pKF);
        }
    }
    // auto end4 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed4 = end4 - start4;
    // timing << "? h_KeyFrames fuse: " << elapsed4.count() << " ms" << std::endl;

    // auto start5 = std::chrono::high_resolution_clock::now();
    checkCudaError(cudaMemcpy(d_MapPoints, h_MapPoints, numValidPoints * sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "Failed to copy h_MapPoints to host");
    checkCudaError(cudaMemcpy(d_KeyFrames, h_KeyFrames, connectedKFSize * sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame), cudaMemcpyHostToDevice), "Failed to copy h_KeyFrames to host");
    checkCudaError(cudaMemcpy(d_Ow, h_Ow, connectedKFSize * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice), "Failed to copy h_Ow to host");
    checkCudaError(cudaMemcpy(d_Tcw, h_Tcw, connectedKFSize * sizeof(Sophus::SE3f), cudaMemcpyHostToDevice), "Failed to copy h_Tcw to host");
    // auto end5 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed5 = end5 - start5;
    // timing << "? cudaMemcpy: " << elapsed5.count() << " ms" << std::endl;

    // auto start6 = std::chrono::high_resolution_clock::now();
    int threads = 256;
    int blocks = (connectedKFSize * numValidPoints + threads - 1) / threads;
    searchAndFuseKernel<<<blocks, threads>>>(d_Ow, d_Tcw, d_KeyFrames, d_MapPoints, 
                                    0, connectedKFSize, 0, numValidPoints, th, 
                                    d_bestDists, d_bestIdxs);

    cudaDeviceSynchronize();
    // auto end6 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed6 = end6 - start6;
    // timing << "? main Kernel: " << elapsed6.count() << " ms" << std::endl;

    // auto start7 = std::chrono::high_resolution_clock::now();
    checkCudaError(cudaMemcpy(bestDists, d_bestDists, numValidPoints * connectedKFSize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDists back to host1");
    checkCudaError(cudaMemcpy(bestIdxs, d_bestIdxs, numValidPoints * connectedKFSize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxs back to host");
    // auto end7 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed7 = end7 - start7;
    // timing << "? cudaMemcpy Back: " << elapsed7.count() << " ms" << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // std::ofstream gpuOutFile("./test/GPU-Side.txt", std::ios::app);    
    // std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
        
    // for (int iKF = 0; iKF < connectedKFSize; iKF++) {
    //     gpuOutFile << "================================= Connected KF: " << connectedKFs[iKF]->mnId << " =================================\n";
    //     cpuOutFile << "================================= Connected KF: " << connectedKFs[iKF]->mnId << " =================================\n";
    //     cpuOutFile.flush();
    //     const set<ORB_SLAM3::MapPoint*> spAlreadyFound = connectedKFs[iKF]->GetMapPoints();
    //     for (int iMP = 0; iMP < numValidPoints; iMP++) {
    //         int idx = iKF*numValidPoints + iMP;
    //         ORB_SLAM3::MapPoint* pMP = vpMapPoints[iMP];

    //         if(spAlreadyFound.count(pMP))
    //             continue;

    //         if (bestDists[idx] != 256)
    //             gpuOutFile << "(i: " << iMP << ", bestDist: " << bestDists[idx] << ", bestIdx: " << bestIdxs[idx] << ")\n";
    //     }
    //     gpuOutFile << "\n\n";
        
    //     origFuse(connectedKFs[iKF], connectedScws[iKF], vpMapPoints, th);
    // }

    // gpuOutFile << "**********************************************************\n";
    // cpuOutFile << "**********************************************************\n";


    // auto start8 = std::chrono::high_resolution_clock::now();
    int nFused = 0;
    for (int iKF = 0; iKF < connectedKFSize; iKF++)
    {
        ORB_SLAM3::KeyFrame* pKF = connectedKFs[iKF];
        const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

        for (int iMP = 0; iMP < numValidPoints; iMP++)
        {
            ORB_SLAM3::MapPoint* pMP = vpMapPoints[iMP];
            if (!pMP || pMP->isBad())
                continue;
            
            int idx = iKF*numValidPoints + iMP;
            int bestDist = bestDists[idx];
            int bestIdx = bestIdxs[idx];

            if (bestDist == 256 || bestIdx == -1)
                continue;
            
            if(spAlreadyFound.count(pMP))
                continue;

            if (bestDist <= TH_LOW) {
                ORB_SLAM3::MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if (pMPinKF) {
                    if (!pMPinKF->isBad()) {
                        vpReplacePoints[iMP] = pMPinKF;
                    }
                }
                else{
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP, bestIdx);
                }
                nFused++;
            }
        }
    }
    // auto end8 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed8 = end8 - start8;
    // timing << "? result: " << elapsed8.count() << " ms" << std::endl;


    return nFused;

}


void SearchAndFuseKernel::origFuse(ORB_SLAM3::KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<ORB_SLAM3::MapPoint*> &vpPoints, const float th)
{
    std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    int validMapPointCounter = -1;
    
    for(int iMP=0; iMP<nPoints; iMP++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];

        if(pMP->isBad())
            continue;
        
        validMapPointCounter++;

        if(spAlreadyFound.count(pMP))
            continue;
        
        // cpuOutFile << "idx: " << validMapPointCounter << ", pMP.mnId: " << pMP->mnId << ", keyframe.mnId: " << pKF->mnId << ", spAlreadyFound.size(): " << spAlreadyFound.size() << std::endl;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if(p3Dc(2)<0.0f)
            continue;

        // Project into Image
        const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

        // Point must be inside the image
        if(!pKF->IsInImage(uv(0),uv(1)))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist3D = PO.norm();

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;

        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++){
            const size_t idx = *vit;

            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel) {
                continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = origDescriptorDistance(dMP,dKF);

            if(dist<bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist != INT_MAX)
            cpuOutFile << "(i: " << validMapPointCounter << ", bestDist: " << bestDist << ", bestIdx: " << bestIdx << ")\n";
    }
    cpuOutFile << "\n\n";
}


int SearchAndFuseKernel::origDescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
