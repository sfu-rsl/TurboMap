#include "Kernels/CudaWrappers/CudaMapPoint.h"
#include "Kernels/CudaKeyFrameDrawer.h"
#include <iostream>
#include <map>
#include <tuple>

#ifdef TIME_MEASURMENT
#define TIMESTAMP_PRINT(msg) std::cout << "TimeStamp [CudaFrame]: " << msg << std::endl
#else
#define TIMESTAMP_PRINT(msg) do {} while (0)
#endif

namespace TRACKING_DATA_WRAPPER
{
    CudaMapPoint::CudaMapPoint() {
        isEmpty = true;
    }

    CudaMapPoint::CudaMapPoint(ORB_SLAM3::MapPoint* mp) {
        isEmpty = false;
        mnId = mp->mnId;
        mbTrackInView = mp->mbTrackInView;
        mbTrackInViewR = mp->mbTrackInViewR;
        mTrackDepth = mp->mTrackDepth;
        mnTrackScaleLevel = mp->mnTrackScaleLevel;
        mTrackViewCos = mp->mTrackViewCos;
        mTrackProjX = mp->mTrackProjX;
        mTrackProjY = mp->mTrackProjY;
        mnTrackScaleLevelR = mp->mnTrackScaleLevelR;
        mTrackViewCosR = mp->mTrackViewCosR;
        mTrackProjXR = mp->mTrackProjXR;
        mTrackProjYR = mp->mTrackProjYR;
        mnLastFrameSeen = mp->mnLastFrameSeen;
        nObs = mp->Observations();
        const cv::Mat& descriptor = mp->GetDescriptor();
        std::memcpy(mDescriptor, descriptor.ptr<uint8_t>(0), descriptor.cols * sizeof(uint8_t));
        mWorldPos = mp->GetWorldPos();
    }
}

namespace MAPPING_DATA_WRAPPER
{
    void CudaMapPoint::initialize() {
        checkCudaError(cudaMalloc((void**)&mObservations_dkf, MAX_NUM_OBSERVATIONS * sizeof(CudaKeyFrame*)), "CudaMapPoint::failed to allocate memory for mObservations_dkf");
    }

    CudaMapPoint::CudaMapPoint() {
        initialize();
        isEmpty = true;
    }

    CudaMapPoint::CudaMapPoint(ORB_SLAM3::MapPoint* mp) {
        isEmpty = false;
        mnId = mp->mnId;
        mbBad = mp->isBad();
        // setObservations(mp);
    }

    void CudaMapPoint::setMemory(ORB_SLAM3::MapPoint* mp) {
        isEmpty = false;
        mnId = mp->mnId;
        mbBad = mp->isBad();
        // setObservations(mp);
    }

    void CudaMapPoint::setObservations(int _nObs, map<ORB_SLAM3::KeyFrame*, tuple<int,int>> observations) {
        nObs = _nObs;
        mObservations_size = observations.size();
        int itr = 0;
        for (const auto& pair : observations) {
            ORB_SLAM3::KeyFrame* key = pair.first;
            const std::tuple<int, int>& value = pair.second; 
            mObservations_leftIdx[itr] = std::get<0>(value);
            mObservations_rightIdx[itr] = std::get<1>(value);

            CudaKeyFrame* d_kf = CudaKeyFrameDrawer::getCudaKeyFrame(key->mnId);
            if (d_kf != nullptr) {
                // mObservations_dkf[itr] = d_kf;
                checkCudaError(cudaMemcpy(&mObservations_dkf[itr], &d_kf, sizeof(MAPPING_DATA_WRAPPER::CudaKeyFrame*), cudaMemcpyHostToDevice), "[CudaMapPoint::setObservations: ] Failed to add d_kf");
            } 
            // else {
            //     cout << "MapPoint Error: KF " << key->mnId << " is not in the drawer.\n";
            // }

            itr++;
        }        
    }

    void CudaMapPoint::freeMemory() {
        checkCudaError(cudaFree(mObservations_dkf),"[CudaMapPoint::] Failed to free memory: mObservations_dkf");
    }
}