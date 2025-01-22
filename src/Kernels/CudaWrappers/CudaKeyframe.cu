#include "Kernels/CudaWrappers/CudaKeyframe.h"

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaKeyframe]: " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

namespace MAPPING_DATA_WRAPPER
{
    void CudaKeyframe::initializeMemory(){
        DEBUG_PRINT("Allocating GPU memory For CudaKeyframe...");

        int nFeatures = CudaUtils::nFeatures_with_th;
        
        bool cameraIsFisheye = CudaUtils::cameraIsFisheye;

        if (cameraIsFisheye) {
            checkCudaError(cudaMalloc((void**)&mvpMapPoints, 2 * nFeatures * sizeof(CudaMapPoint)), "CudaKeyframe::failed to allocate memory for mvpMapPoints");
        } else {
            checkCudaError(cudaMalloc((void**)&mvpMapPoints, nFeatures * sizeof(CudaMapPoint)), "CudaKeyframe::failed to allocate memory for mvpMapPoints");
        }

        checkCudaError(cudaMalloc((void**)&mvKeys, nFeatures * sizeof(CudaKeyPoint)), "CudaKeyframe::failed to allocate memory for mvKeys");
        
        checkCudaError(cudaMalloc((void**)&mvKeysRight, nFeatures * sizeof(CudaKeyPoint)), "CudaKeyframe::failed to allocate memory for mvKeysRight");
        
        checkCudaError(cudaMalloc((void**)&mvKeysUn, nFeatures * sizeof(CudaKeyPoint)), "CudaKeyframe::failed to allocate memory for mvKeysUn"); 
    }

    CudaKeyframe::CudaKeyframe() {
        initializeMemory();
    }

    void CudaKeyframe::setMemory(ORB_SLAM3::KeyFrame* KF) {
        DEBUG_PRINT("Filling CudaKeyframe Memory With Keyframe Data...");

        Nleft = KF -> NLeft;
        mnId = KF -> mnId;

        std::vector<CudaKeyPoint> tmp_mvKeys(mvKeys_size);
        for (int i = 0; i < mvKeys_size; ++i){
            tmp_mvKeys[i].ptx = KF->mvKeys[i].pt.x;
            tmp_mvKeys[i].pty = KF->mvKeys[i].pt.y;
            tmp_mvKeys[i].octave = KF->mvKeys[i].octave;
        }
        checkCudaError(cudaMemcpy((void*) mvKeys, tmp_mvKeys.data(), mvKeys_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyframe:: Failed to copy mvKeys to gpu");

        std::vector<CudaKeyPoint> tmp_mvKeysRight(mvKeysRight_size);        
        for (int i = 0; i < mvKeysRight_size; ++i){
            tmp_mvKeysRight[i].ptx = KF->mvKeysRight[i].pt.x;
            tmp_mvKeysRight[i].pty = KF->mvKeysRight[i].pt.y;
            tmp_mvKeysRight[i].octave = KF->mvKeysRight[i].octave;
        }
        checkCudaError(cudaMemcpy((void*) mvKeysRight, tmp_mvKeysRight.data(), mvKeysRight_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyframe:: Failed to copy mvKeysRight to gpu");

        std::vector<CudaKeyPoint> tmp_mvKeysUn(mvKeysUn_size);   
        for (int i = 0; i < mvKeysUn_size; ++i){
            tmp_mvKeysUn[i].ptx = KF->mvKeysUn[i].pt.x;
            tmp_mvKeysUn[i].pty = KF->mvKeysUn[i].pt.y;
            tmp_mvKeysUn[i].octave = KF->mvKeysUn[i].octave;
        }
        checkCudaError(cudaMemcpy(mvKeysUn, tmp_mvKeysUn.data(), mvKeysUn_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyframe:: Failed to copy mvKeysUn to gpu");

        vector<ORB_SLAM3::MapPoint*> h_mapPoints = KF->GetMapPointMatches();
        mvpMapPoints_size = h_mapPoints.size();
        std::vector<CudaMapPoint> tmp_mvpMapPoints(mvpMapPoints_size);
        for (int i = 0; i < mvpMapPoints_size; ++i) {
                if (h_mapPoints[i]) {
                    CudaMapPoint cuda_mp(h_mapPoints[i]);
                    tmp_mvpMapPoints[i] = cuda_mp;
                } else {
                    CudaMapPoint cuda_mp;
                    tmp_mvpMapPoints[i] = cuda_mp;            
                }
        }
        checkCudaError(cudaMemcpy(mvpMapPoints, tmp_mvpMapPoints.data(), tmp_mvpMapPoints.size() * sizeof(CudaMapPoint), cudaMemcpyHostToDevice), "CudaKeyframe:: Failed to copy mvpMapPoints to gpu");
    }

    void CudaKeyframe::freeMemory(){
        DEBUG_PRINT("Freeing GPU Memory For CudaKeyframe...");
        checkCudaError(cudaFree(mvpMapPoints),"[CudaKeyframe::] Failed to free memory: mvpMapPoints");
        checkCudaError(cudaFree((void*) mvKeys),"[CudaKeyframe::] Failed to free memory: mvKeys");
        checkCudaError(cudaFree((void*) mvKeysRight),"[CudaKeyframe::] Failed to free memory: mvKeysRight");
        checkCudaError(cudaFree(mvKeysUn),"[CudaKeyframe::] Failed to free memory: mvKeysUn");
    }

}