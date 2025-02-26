#include "Kernels/CudaWrappers/CudaKeyFrame.h"
#include "Kernels/CudaMapPointStorage.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaKeyFrame]: " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

namespace MAPPING_DATA_WRAPPER
{
    void CudaKeyFrame::initializeMemory(){
        DEBUG_PRINT("Allocating GPU memory For CudaKeyFrame...");

        int nFeatures = CudaUtils::nFeatures_with_th;
        
        bool cameraIsFisheye = CudaUtils::cameraIsFisheye;

        if (cameraIsFisheye) {
            checkCudaError(cudaMalloc((void**)&mvpMapPoints, 2 * nFeatures * sizeof(CudaMapPoint*)), "CudaKeyFrame::failed to allocate memory for mvpMapPoints");
            h_mvpMapPoints.resize(2 * nFeatures, nullptr);
            mvpMapPoints_size = 2 * nFeatures;
        } else {
            checkCudaError(cudaMalloc((void**)&mvpMapPoints, nFeatures * sizeof(CudaMapPoint*)), "CudaKeyFrame::failed to allocate memory for mvpMapPoints");
            h_mvpMapPoints.resize(nFeatures, nullptr);
            mvpMapPoints_size = nFeatures;
        }

        if (cameraIsFisheye) {
            checkCudaError(cudaMalloc((void**)&mvDepth, 2 * nFeatures * sizeof(float)), "Frame::failed to allocate memory for mvDepth");
        } else {
            checkCudaError(cudaMalloc((void**)&mvDepth, nFeatures * sizeof(float)), "Frame::failed to allocate memory for mvDepth");
        }

        checkCudaError(cudaMalloc((void**)&mvKeys, nFeatures * sizeof(CudaKeyPoint)), "CudaKeyFrame::failed to allocate memory for mvKeys");
        
        checkCudaError(cudaMalloc((void**)&mvKeysRight, nFeatures * sizeof(CudaKeyPoint)), "CudaKeyFrame::failed to allocate memory for mvKeysRight");
        
        checkCudaError(cudaMalloc((void**)&mvKeysUn, nFeatures * sizeof(CudaKeyPoint)), "CudaKeyFrame::failed to allocate memory for mvKeysUn"); 
    }

    CudaKeyFrame::CudaKeyFrame() {
        isEmpty = true;
        initializeMemory();
    }

    void CudaKeyFrame::setGPUAddress(CudaKeyFrame* ptr) {
        gpuAddr = ptr;
    }

    void CudaKeyFrame::setMemory(ORB_SLAM3::KeyFrame* KF) {
        DEBUG_PRINT("Filling CudaKeyFrame Memory With Keyframe Data...");

        isEmpty = false;
        NLeft = KF->NLeft;
        mnId = KF->mnId;
        mThDepth = KF->mThDepth;
        
        checkCudaError(cudaMemcpy(mvDepth, KF->mvDepth.data(), KF->mvDepth.size() * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvDepth to gpu");

        mvKeys_size = KF->mvKeys.size();
        std::vector<CudaKeyPoint> tmp_mvKeys(mvKeys_size);
        for (int i = 0; i < mvKeys_size; ++i){
            tmp_mvKeys[i].ptx = KF->mvKeys[i].pt.x;
            tmp_mvKeys[i].pty = KF->mvKeys[i].pt.y;
            tmp_mvKeys[i].octave = KF->mvKeys[i].octave;
        }
        checkCudaError(cudaMemcpy((void*) mvKeys, tmp_mvKeys.data(), mvKeys_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeys to gpu");

        mvKeysRight_size = KF->mvKeysRight.size();
        std::vector<CudaKeyPoint> tmp_mvKeysRight(mvKeysRight_size);        
        for (int i = 0; i < mvKeysRight_size; ++i){
            tmp_mvKeysRight[i].ptx = KF->mvKeysRight[i].pt.x;
            tmp_mvKeysRight[i].pty = KF->mvKeysRight[i].pt.y;
            tmp_mvKeysRight[i].octave = KF->mvKeysRight[i].octave;
        }
        checkCudaError(cudaMemcpy((void*) mvKeysRight, tmp_mvKeysRight.data(), mvKeysRight_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeysRight to gpu");

        mvKeysUn_size = KF->mvKeysUn.size();
        std::vector<CudaKeyPoint> tmp_mvKeysUn(mvKeysUn_size);   
        for (int i = 0; i < mvKeysUn_size; ++i){
            tmp_mvKeysUn[i].ptx = KF->mvKeysUn[i].pt.x;
            tmp_mvKeysUn[i].pty = KF->mvKeysUn[i].pt.y;
            tmp_mvKeysUn[i].octave = KF->mvKeysUn[i].octave;
        }
        checkCudaError(cudaMemcpy(mvKeysUn, tmp_mvKeysUn.data(), mvKeysUn_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeysUn to gpu");

        vector<ORB_SLAM3::MapPoint*> h_mapPoints = KF->GetMapPointMatches();
        mvpMapPoints_size = h_mapPoints.size();
        CudaMapPoint* d_mp;
        for (int i = 0; i < mvpMapPoints_size; ++i) {
            if (h_mapPoints[i]) {
                d_mp = CudaMapPointStorage::getCudaMapPoint(h_mapPoints[i]->mnId);
            } else {
                d_mp = nullptr;
            }
            checkCudaError(cudaMemcpy(&mvpMapPoints[i], &d_mp, sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint*), cudaMemcpyHostToDevice), "[CudaKeyFrame::setMvpMapPoints: ] Failed to copy mvpmapPoints");
        }
    }

    void CudaKeyFrame::addMapPoint(MAPPING_DATA_WRAPPER::CudaMapPoint* d_mp, int idx) {
        checkCudaError(cudaMemcpy(&mvpMapPoints[idx], &d_mp, sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint*), cudaMemcpyHostToDevice), "[CudaKeyFrame::updateMvpMapPoints: ] Failed to update idx ");
    }

    void CudaKeyFrame::eraseMapPoint(int idx) {
        CudaMapPoint* d_mp = nullptr;
        checkCudaError(cudaMemcpy(&mvpMapPoints[idx], &d_mp, sizeof(MAPPING_DATA_WRAPPER::CudaMapPoint*), cudaMemcpyHostToDevice), "[CudaKeyFrame::updateMvpMapPoints: ] Failed to update idx ");
    }

    void CudaKeyFrame::freeMemory(){
        DEBUG_PRINT("Freeing GPU Memory For CudaKeyFrame...");
        checkCudaError(cudaFree(mvpMapPoints),"[CudaKeyFrame::] Failed to free memory: mvpMapPoints");
        checkCudaError(cudaFree((void*) mvKeys),"[CudaKeyFrame::] Failed to free memory: mvKeys");
        checkCudaError(cudaFree((void*) mvKeysRight),"[CudaKeyFrame::] Failed to free memory: mvKeysRight");
        checkCudaError(cudaFree(mvKeysUn),"[CudaKeyFrame::] Failed to free memory: mvKeysUn");
    }
}