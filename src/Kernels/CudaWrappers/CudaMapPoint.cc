#include "Kernels/CudaWrappers/CudaMapPoint.h"
#include "Kernels/CudaKeyframeDrawer.h"
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
    CudaMapPoint::CudaMapPoint() {
        isEmpty = true;
    }

    CudaMapPoint::CudaMapPoint(ORB_SLAM3::MapPoint* mp) {
        isEmpty = false;
        mnId = mp->mnId;
        mbBad = mp->isBad();
        nObs = mp->Observations();
        const map<ORB_SLAM3::KeyFrame*, tuple<int,int>> observations = mp->GetObservations();
        mObservations_size = observations.size();
        for (const auto& pair : observations) {
            ORB_SLAM3::KeyFrame* key = pair.first;
            const std::tuple<int, int>& value = pair.second;  

            CudaKeyframe* d_kf = CudaKeyframeDrawer::getCudaKeyframe(key->mnId);
            if (d_kf == nullptr) {
                d_kf = CudaKeyframeDrawer::addCudaKeyframe(key);
            }
            Observation obs = Observation(d_kf, std::get<0>(value), std::get<1>(value));
            mObservations.push_back(obs);
        }
    }
}