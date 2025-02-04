#ifndef CUDA_MAPPOINT_H
#define CUDA_MAPPOINT_H

#include "MapPoint.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

namespace TRACKING_DATA_WRAPPER {

class CudaMapPoint {
    public:
        CudaMapPoint();
        CudaMapPoint(ORB_SLAM3::MapPoint* mp);
    
    public:
        // For creating empty mapPoints instead of using null ptr
        bool isEmpty;

    // For searchByProjection in TrackLocalMap
    public:
        long unsigned int mnId;
        bool mbBad;
        bool mbTrackInView;
        bool mbTrackInViewR;
        float mTrackDepth;
        int mnTrackScaleLevel;
        float mTrackViewCos;
        float mTrackProjX;
        float mTrackProjY;
        int mnTrackScaleLevelR;
        float mTrackViewCosR;
        float mTrackProjXR;
        float mTrackProjYR;
        long unsigned int mnLastFrameSeen;
        int nObs;
        uint8_t mDescriptor[32];
        
    // For searchByProjection in PoseEstimation
    public:
        Eigen::Vector3f mWorldPos;
    };
}

namespace MAPPING_DATA_WRAPPER {

class CudaKeyframe;

// struct Observation {
//     CudaKeyframe* d_kf;
//     int leftIndex;
//     int rightIndex;

//     Observation() {}

//     Observation(CudaKeyframe* d_kf, int leftIndex, int rightIndex) 
//     : d_kf(d_kf), leftIndex(leftIndex), rightIndex(rightIndex) {}
// };

class CudaMapPoint {
    public:
        CudaMapPoint();
        CudaMapPoint(ORB_SLAM3::MapPoint* mp);
        CudaMapPoint(ORB_SLAM3::MapPoint* mp, long unsigned int _observerId, CudaKeyframe* d_kf);
        
        // void setMbBad(bool _mbBad) {mbBad = _mbBad;};
        // void setNObs(int _nObs) {nObs = _nObs;}; 
        // void setObserver(long unsigned int _observerId, CudaKeyframe* d_kf) {observerId = _observerId; observer = d_kf;};

        void setObservations(ORB_SLAM3::MapPoint* mp);

    public:
        // For creating empty mapPoints instead of using null ptr
        bool isEmpty;

    public:
        long unsigned int mnId;
        bool mbBad;
        int nObs;
        int mObservations_size;
        CudaKeyframe* mObservations_dkf[200];
        int mObservations_leftIdx[200];
        int mObservations_rightIdx[200];
        // Observation mObservations[200];
        CudaKeyframe* observer;
        long unsigned int observerId;
    };
}

#endif // CUDA_MAPPOINT_H