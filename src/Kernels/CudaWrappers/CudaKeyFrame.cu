#include "Kernels/CudaWrappers/CudaKeyFrame.h"
#include <cstdio>
#include <vector>

// #define DEBUG
// #define TIME_MEASURMENT

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaFrame]: " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

#ifdef TIME_MEASURMENT
#define TIMESTAMP_PRINT(msg) std::cout << "TimeStamp [CudaFrame]: " << msg << std::endl
#else
#define TIMESTAMP_PRINT(msg) do {} while (0)
#endif


namespace MAPPING_DATA_WRAPPER
{
    void CudaKeyFrame::initializeMemory(){
        DEBUG_PRINT("Allocating GPU memory For KeyFrame...");

        int nFeatures = CudaUtils::nFeatures_with_th;
                
        bool cameraIsFisheye = CudaUtils::cameraIsFisheye;

        checkCudaError(cudaMalloc((void**)&mvScaleFactors, nFeatures * sizeof(float)), "KeyFrame::failed to allocate memory for mvScaleFactors");

        checkCudaError(cudaMalloc((void**)&mvuRight, nFeatures * sizeof(float)), "KeyFrame::failed to allocate memory for mvuRight");

        checkCudaError(cudaMalloc((void**)&mvKeys, nFeatures * sizeof(CudaKeyPoint)), "KeyFrame::failed to allocate memory for mvKeys");
        
        checkCudaError(cudaMalloc((void**)&mvKeysRight, nFeatures * sizeof(CudaKeyPoint)), "KeyFrame::failed to allocate memory for mvKeysRight");
        
        checkCudaError(cudaMalloc((void**)&mvKeysUn, nFeatures * sizeof(CudaKeyPoint)), "KeyFrame::failed to allocate memory for mvKeysUn"); 
        
        checkCudaError(cudaMalloc((void**)&mvInvLevelSigma2, nFeatures * sizeof(float)), "KeyFrame::failed to allocate memory for mvInvLevelSigma2");

        if (cameraIsFisheye) {
            checkCudaError(cudaMalloc((void**)&mDescriptors, 2 * nFeatures * DESCRIPTOR_SIZE * sizeof(uint8_t)), "Frame::failed to allocate memory for mDescriptors");
        } else {
            checkCudaError(cudaMalloc((void**)&mDescriptors, nFeatures * DESCRIPTOR_SIZE * sizeof(uint8_t)), "Frame::failed to allocate memory for mDescriptors");
        }
    }

    CudaKeyFrame::CudaKeyFrame() {
        initializeMemory();
    }

    void CudaKeyFrame::setMvKeys(CudaKeyPoint* const &_mvKeys) {
         mvKeys = _mvKeys; 
         mvKeysIsOnGpu = true;
    }
    
    void CudaKeyFrame::setMvKeysRight(CudaKeyPoint* const &_mvKeysRight) {
        mvKeysRight = _mvKeysRight; 
        mvKeysRightIsOnGpu = true;
    }

    void CudaKeyFrame::setMDescriptors(uint8_t* const &_mDescriptors) { 
        mDescriptors = _mDescriptors; 
        mDescriptorsIsOnGpu = true;
    }

    void CudaKeyFrame::setMemory(const ORB_SLAM3::KeyFrame &KF) {
        DEBUG_PRINT("Filling CudaKeyFrame Memory With KeyFrame Data...");

        mnId = KF.mnId;
        fx = KF.fx;
        fy = KF.fy;
        cx = KF.cx;
        cy = KF.cy;
        mbf = KF.mbf;
        Nleft = F.Nleft;

        mvScaleFactors_size = KF.mvScaleFactors.size();
        mvKeys_size = KF.mvKeys.size();
        mvKeysRight_size = KF.mvKeysRight.size();
        mvKeysUn_size = KF.mvKeysUn.size();
        mvuRight_size = KF.mvuRight.size();
        mvInvLevelSigma2_size = KF.mvInvLevelSigma2.size();
        mDescriptor_rows = KF.mDescriptors.rows;

        checkCudaError(cudaMemcpy(mvScaleFactors, KF.mvScaleFactors.data(), mvScaleFactors_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvScaleFactors to gpu");

        checkCudaError(cudaMemcpy(mvInvLevelSigma2, KF.mvInvLevelSigma2.data(), mvInvLevelSigma2_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvInvLevelSigma2 to gpu");

        checkCudaError(cudaMemcpy(mvuRight, KF.mvuRight.data(), mvuRight_size * sizeof(float), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvuRight to gpu");
        
        if (!mDescriptorsIsOnGpu) {
            checkCudaError(cudaMemcpy((void*) mDescriptors, KF.mDescriptors.data,  KF.mDescriptors.rows * DESCRIPTOR_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mDescriptors to gpu"); 
        } 
        
        if (!mvKeysIsOnGpu) {
            std::vector<CudaKeyPoint> tmp_mvKeys(mvKeys_size);
            for (int i = 0; i < mvKeys_size; ++i){
                tmp_mvKeys[i].ptx = KF.mvKeys[i].pt.x;
                tmp_mvKeys[i].pty = KF.mvKeys[i].pt.y;
                tmp_mvKeys[i].octave = KF.mvKeys[i].octave;
            }
            checkCudaError(cudaMemcpy((void*) mvKeys, tmp_mvKeys.data(), mvKeys_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeys to gpu");
        }

        if (!mvKeysRightIsOnGpu) {
            std::vector<CudaKeyPoint> tmp_mvKeysRight(mvKeysRight_size);        
            for (int i = 0; i < mvKeysRight_size; ++i){
                tmp_mvKeysRight[i].ptx = KF.mvKeysRight[i].pt.x;
                tmp_mvKeysRight[i].pty = KF.mvKeysRight[i].pt.y;
                tmp_mvKeysRight[i].octave = KF.mvKeysRight[i].octave;
            }
            checkCudaError(cudaMemcpy((void*) mvKeysRight, tmp_mvKeysRight.data(), mvKeysRight_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeysRight to gpu");
        }

        std::vector<CudaKeyPoint> tmp_mvKeysUn(mvKeysUn_size);   
        for (int i = 0; i < mvKeysUn_size; ++i){
            tmp_mvKeysUn[i].ptx = KF.mvKeysUn[i].pt.x;
            tmp_mvKeysUn[i].pty = KF.mvKeysUn[i].pt.y;
            tmp_mvKeysUn[i].octave = KF.mvKeysUn[i].octave;
        }
        checkCudaError(cudaMemcpy(mvKeysUn, tmp_mvKeysUn.data(), mvKeysUn_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaKeyFrame:: Failed to copy mvKeysUn to gpu");

        // std::memcpy(mpCamera_mvParameters, KF.mpCamera->getParameters().data(), F.mpCamera->getParameters().size() * sizeof(float));
        
        checkCudaError(cudaDeviceSynchronize(), "[cudaKeyFrame:] failed to set memory");  

    }

    void CudaKeyFrame::freeMemory(){
        DEBUG_PRINT("Freeing GPU Memory For KeyFrame...");
        checkCudaError(cudaFree(mvScaleFactors),"Failed to free keyframe memory: mvScaleFactors");
        checkCudaError(cudaFree(mvuRight),"Failed to free keyframe memory: mvInvLevelSigma2");
        checkCudaError(cudaFree(mvuRight),"Failed to free keyframe memory: mvuRight");
        if (!mDescriptorsIsOnGpu)
            checkCudaError(cudaFree((void*) mDescriptors),"Failed to free keyframe memory: mDescriptors");
        if (!mvKeysIsOnGpu)   
            checkCudaError(cudaFree((void*) mvKeys),"Failed to free keyframe memory: mvKeys");
        if (!mvKeysRightIsOnGpu)
            checkCudaError(cudaFree((void*) mvKeysRight),"Failed to free keyframe memory: mvKeysRight");
        checkCudaError(cudaFree(mvKeysUn),"Failed to free keyframe memory: mvKeysUn");
    }
}