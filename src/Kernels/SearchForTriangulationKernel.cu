#include <iostream>
#include <fstream>

#include "Kernels/SearchForTriangulationKernel.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "sophus/sim3.hpp"
#include <cusolverDn.h>
#include <Eigen/Dense>
#include <math.h>

// It is advisable to define the default dense index type as int for CUDA device code.
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

// Mark the function as __host__ __device__ so that it can be used on both sides if needed.
__host__ __device__ inline Eigen::Matrix3f deviceInverse(const Eigen::Matrix3f& M) {
    // Unpack matrix elements
    float a = M(0,0), b = M(0,1), c = M(0,2);
    float d = M(1,0), e = M(1,1), f = M(1,2);
    float g = M(2,0), h = M(2,1), i = M(2,2);
    
    // Compute determinant
    float det = a * (e * i - f * h)
              - b * (d * i - f * g)
              + c * (d * h - e * g);
              
    // Handle singular matrix (here we simply return Identity)
    if (fabsf(det) < 1e-6f) {
        return Eigen::Matrix3f::Identity();
    }
    
    float invDet = 1.0f / det;
    
    Eigen::Matrix3f inv;
    inv(0,0) =  (e * i - f * h) * invDet;
    inv(0,1) = -(b * i - c * h) * invDet;
    inv(0,2) =  (b * f - c * e) * invDet;
    inv(1,0) = -(d * i - f * g) * invDet;
    inv(1,1) =  (a * i - c * g) * invDet;
    inv(1,2) = -(a * f - c * d) * invDet;
    inv(2,0) =  (d * h - e * g) * invDet;
    inv(2,1) = -(a * h - b * g) * invDet;
    inv(2,2) =  (a * e - b * d) * invDet;
    
    return inv;
}

__device__ bool pinholeEpipolarConstrain(
    MAPPING_DATA_WRAPPER::CudaCamera camera1, MAPPING_DATA_WRAPPER::CudaCamera camera2, const TRACKING_DATA_WRAPPER::CudaKeyPoint &kp1, const TRACKING_DATA_WRAPPER::CudaKeyPoint &kp2, 
    const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12, const float sigmaLevel, const float unc) {
    //Compute Fundamental Matrix
    Eigen::Matrix3f t12x = Sophus::SO3f::hat(t12);
    Eigen::Matrix3f K1 = camera1.toK;
    Eigen::Matrix3f K2 = camera2.toK;
    Eigen::Matrix3f F12 = deviceInverse(K1.transpose()) * t12x * R12 * deviceInverse(K2);
    
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.ptx*F12(0,0) + kp1.pty*F12(1,0) + F12(2,0);
    const float b = kp1.ptx*F12(0,1) + kp1.pty*F12(1,1) + F12(2,1);
    const float c = kp1.ptx*F12(0,2) + kp1.pty*F12(1,2) + F12(2,2);

    const float num = a*kp2.ptx+b*kp2.pty+c;

    const float den = a*a + b*b;

    if (den == 0)
        return false;

    const float dsqr = num*num / den;

    return dsqr < 3.84*unc;
}

// __device__ function that computes (approximately) the nullspace vector of A.
// A is a 4x4 matrix (by value, in row–major order as stored in Eigen) and x3D is returned by reference.
__device__ void solveSVD(Eigen::Matrix<float,4,4> A, Eigen::Vector3f &x3D) {
    // Compute symmetric matrix M = Aᵀ * A.
    Eigen::Matrix<float,4,4> M = A.transpose() * A;
    
    // Initialize V to the 4x4 identity (accumulates Jacobi rotations)
    Eigen::Matrix<float,4,4> V = Eigen::Matrix<float,4,4>::Identity();
    
    const int maxIter = 50;
    const float eps = 1e-6f;
    
    // Perform Jacobi rotations on M.
    for (int iter = 0; iter < maxIter; iter++) {
        // Find the largest off-diagonal element in M (upper triangle).
        int p = 0, q = 1;
        float maxOff = fabsf(M(0,1));
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                float off = fabsf(M(i, j));
                if (off > maxOff) {
                    maxOff = off;
                    p = i;
                    q = j;
                }
            }
        }
        if (maxOff < eps) break;  // Converged.
        
        float app = M(p, p);
        float aqq = M(q, q);
        float apq = M(p, q);
        
        // Compute rotation angle theta.
        float theta = 0.5f * atanf((2.0f * apq) / (aqq - app));
        float c = cosf(theta);
        float s = sinf(theta);
        
        // Rotate rows/columns p and q of M.
        for (int k = 0; k < 4; k++) {
            if (k == p || k == q) continue;
            float mkp = M(k, p);
            float mkq = M(k, q);
            float new_mkp = c * mkp - s * mkq;
            float new_mkq = s * mkp + c * mkq;
            M(k, p) = new_mkp;
            M(p, k) = new_mkp;  // enforce symmetry
            M(k, q) = new_mkq;
            M(q, k) = new_mkq;
        }
        // Update the diagonal entries.
        float new_app = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        float new_aqq = s * s * app + 2.0f * s * c * apq + c * c * aqq;
        M(p, p) = new_app;
        M(q, q) = new_aqq;
        M(p, q) = 0.0f;
        M(q, p) = 0.0f;
        
        // Update the eigenvector matrix V.
        for (int i = 0; i < 4; i++) {
            float vip = V(i, p);
            float viq = V(i, q);
            V(i, p) = c * vip - s * viq;
            V(i, q) = s * vip + c * viq;
        }
    }
    
    // Find the index of the smallest eigenvalue (the smallest diagonal element in M).
    int minIndex = 0;
    float minEigen = M(0, 0);
    for (int i = 1; i < 4; i++) {
        if (M(i, i) < minEigen) {
            minEigen = M(i, i);
            minIndex = i;
        }
    }
    
    // The corresponding eigenvector (4D) is the minIndex-th column of V.
    Eigen::Vector4f v = V.col(minIndex);
    
    // Convert from homogeneous (4D) to Euclidean (3D) coordinates:
    if (fabsf(v(3)) < eps) {
        x3D = v.head<3>(); // Avoid division if nearly zero.
    } else {
        x3D = v.head<3>() / v(3);
    }
}

__device__ void triangulate(float &p1x, float &p1y, float &p2x, float &p2y, const Eigen::Matrix<float,3,4> &Tcw1,
                            const Eigen::Matrix<float,3,4> &Tcw2, Eigen::Vector3f &x3D) {
    Eigen::Matrix<float,4,4> A;
    A.row(0) = p1x*Tcw1.row(2)-Tcw1.row(0);
    A.row(1) = p1y*Tcw1.row(2)-Tcw1.row(1);
    A.row(2) = p2x*Tcw2.row(2)-Tcw2.row(0);
    A.row(3) = p2y*Tcw2.row(2)-Tcw2.row(1);

    solveSVD(A, x3D);

    // Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
    // Eigen::Vector4f x3Dh = svd.matrixV().col(3);
    // x3D = x3Dh.head(3)/x3Dh(3);
}

__device__ Eigen::Vector2f project(const Eigen::Vector3f &v3D, float* mvParameters) {
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
}

__device__ Eigen::Vector3f unprojectEig(float p2Dx, float p2Dy, float* mvParameters, const float camPrecision) {

    //Use Newthon method to solve for theta with good precision (err ~ e-6)
    float pwx = (p2Dx - mvParameters[2]) / mvParameters[0];
    float pwy = (p2Dy - mvParameters[3]) / mvParameters[1];
    float scale = 1.f;
    float theta_d = sqrtf(pwx*pwx + pwy*pwy);
    theta_d = fminf(fmaxf(-CV_PI / 2.f, theta_d), CV_PI / 2.f);

    if (theta_d > 1e-8) {
        //Compensate distortion iteratively
        float theta = theta_d;

        for (int j = 0; j < 10; j++) {
            float theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 =
                    theta4 * theta4;
            float k0_theta2 = mvParameters[4] * theta2, k1_theta4 = mvParameters[5] * theta4;
            float k2_theta6 = mvParameters[6] * theta6, k3_theta8 = mvParameters[7] * theta8;
            float theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
            theta = theta - theta_fix;
            if (fabsf(theta_fix) < camPrecision)
                break;
        }
        //scale = theta - theta_d;
        scale = std::tan(theta) / theta_d;
    }

    return Eigen::Vector3f(pwx*scale, pwy*scale, 1.f);
}

__device__ float triangulateMatches(const TRACKING_DATA_WRAPPER::CudaKeyPoint &kp1, const TRACKING_DATA_WRAPPER::CudaKeyPoint &kp2, float *mvParameters1, float *mvParameters2, 
                                    const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12, const float sigmaLevel, const float unc, const float camPrecision, 
                                    Eigen::Vector3f& p3D) {

    Eigen::Vector3f r1 = unprojectEig(kp1.ptx, kp1.pty, mvParameters1, camPrecision);
    Eigen::Vector3f r2 = unprojectEig(kp2.ptx, kp2.pty, mvParameters2, camPrecision);

    //Check parallax
    Eigen::Vector3f r21 = R12 * r2;

    const float cosParallaxRays = r1.dot(r21) / (r1.norm()*r21.norm());

    if(cosParallaxRays > 0.9998){
        return -1;
    }

    //Parallax is good, so we try to triangulate
    float p11x, p11y, p22x, p22y;

    p11x = r1[0];
    p11y = r1[1];

    p22x = r2[0];
    p22y = r2[1];

    Eigen::Vector3f x3D;
    Eigen::Matrix<float,3,4> Tcw1;
    Tcw1 << Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero();

    Eigen::Matrix<float,3,4> Tcw2;

    Eigen::Matrix3f R21 = R12.transpose();
    Tcw2 << R21, -R21 * t12;


    triangulate(p11x, p11y, p22x, p22y, Tcw1, Tcw2, x3D);
    // cv::Mat x3Dt = x3D.t();

    float z1 = x3D(2);
    if (z1 <= 0) {
        return -2;
    }

    float z2 = R21.row(2).dot(x3D)+Tcw2(2,3);
    if (z2<=0) {
        return -3;
    }

    //Check reprojection error
    Eigen::Vector2f uv1 = project(x3D, mvParameters1);

    float errX1 = uv1(0) - kp1.ptx;
    float errY1 = uv1(1) - kp1.pty;

    if ((errX1*errX1+errY1*errY1) > 5.991*sigmaLevel){   //Reprojection error is high
        return -4;
    }

    Eigen::Vector3f x3D2 = R21 * x3D + Tcw2.col(3);
    Eigen::Vector2f uv2 = project(x3D2, mvParameters2);

    float errX2 = uv2(0) - kp2.ptx;
    float errY2 = uv2(1) - kp2.pty;

    if ((errX2*errX2+errY2*errY2) > 5.991*unc){   //Reprojection error is high
        return -5;
    }

    p3D = x3D;

    return z1;
}

__device__ bool fisheyeEpipolarConstrain(MAPPING_DATA_WRAPPER::CudaCamera camera1, MAPPING_DATA_WRAPPER::CudaCamera camera2, 
                                         const TRACKING_DATA_WRAPPER::CudaKeyPoint &kp1, const TRACKING_DATA_WRAPPER::CudaKeyPoint &kp2,
                                         const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12, const float sigmaLevel, 
                                         const float unc, const float camPrecision) {
    Eigen::Vector3f p3D;
    return triangulateMatches(kp1, kp2, camera1.mvParameters, camera2.mvParameters, R12, t12, sigmaLevel, unc, camPrecision, p3D) > 0.0001f;
}

__global__ void searchForTriangulationKernel(
    size_t featVecSize, size_t keyFrameMapPointCount, size_t keyFrameMaxFeatureCount, const float scaleFactor, const float camPrecision, const bool bCoarse,
    size_t *currFrameFeatVecIdxCorrespondences, size_t *neighFramesFeatVecIdxCorrespondences,
    size_t *currFrameFeatVec, size_t *currFrameFeatVecIdxs, size_t *neighFramesfeatVec, size_t *neighFramesfeatVecIdxs, size_t *neighFramesFeatVecStartIdxs,
    size_t currFrameNLeft, size_t *neighFramesNLeft,
    Eigen::Matrix3f *Rll, Eigen::Matrix3f *Rlr, Eigen::Matrix3f *Rrl, Eigen::Matrix3f *Rrr, 
    Eigen::Vector3f *tll, Eigen::Vector3f *tlr, Eigen::Vector3f *trl, Eigen::Vector3f *trr, 
    bool *currFrameMapPointExists, bool *neighFramesMapPointExists, 
    uchar *currFrameDescriptors, uchar *neighFramesDescriptors, 
    TRACKING_DATA_WRAPPER::CudaKeyPoint *currFrameMvKeysUn, TRACKING_DATA_WRAPPER::CudaKeyPoint *neighFramesMvKeysUn, 
    TRACKING_DATA_WRAPPER::CudaKeyPoint *currFrameMvKeys, TRACKING_DATA_WRAPPER::CudaKeyPoint *neighFramesMvKeys, 
    TRACKING_DATA_WRAPPER::CudaKeyPoint *currFrameMvKeysRight, TRACKING_DATA_WRAPPER::CudaKeyPoint *neighFramesMvKeysRight,
    float *currFrameMvuRight, float *neighFramesMvuRight,
    MAPPING_DATA_WRAPPER::CudaCamera currFrameCamera1, MAPPING_DATA_WRAPPER::CudaCamera currFrameCamera2, 
    MAPPING_DATA_WRAPPER::CudaCamera *neighFramesCamera1, MAPPING_DATA_WRAPPER::CudaCamera *neighFramesCamera2,
    Eigen::Vector2f *ep,
    size_t *matchedPairIndexes
    ) {

    int neighborIdx = blockIdx.x;
    int correspondingFeatVecIdx = blockIdx.y;

    Eigen::Matrix3f R12;
    Eigen::Vector3f t12;

    int currFrameFeatVecCorrespondenceIdx = currFrameFeatVecIdxCorrespondences[neighborIdx*featVecSize + correspondingFeatVecIdx];
    int neighFrameFeatVecCorrespondenceIdx = neighFramesFeatVecIdxCorrespondences[neighborIdx*featVecSize + correspondingFeatVecIdx];

    size_t currFeatVecCount = (currFrameFeatVecCorrespondenceIdx == 0) ? currFrameFeatVecIdxs[currFrameFeatVecCorrespondenceIdx] 
                                               : currFrameFeatVecIdxs[currFrameFeatVecCorrespondenceIdx] - currFrameFeatVecIdxs[currFrameFeatVecCorrespondenceIdx - 1];
    int currFeatStartIdx = (currFrameFeatVecCorrespondenceIdx == 0) ? 0 : currFrameFeatVecIdxs[currFrameFeatVecCorrespondenceIdx - 1];

    for (size_t i1 = 0; i1 < currFeatVecCount; i1++) {

        const size_t idx1 = currFrameFeatVec[currFeatStartIdx + i1];
        matchedPairIndexes[neighborIdx*featVecSize + idx1] = -1;

        if (currFrameMapPointExists[idx1])
            return;

        const bool bStereo1 = (!currFrameCamera2.isAvailable && currFrameMvuRight[idx1] >= 0);

        const TRACKING_DATA_WRAPPER::CudaKeyPoint &kp1 = (currFrameNLeft == -1) ? currFrameMvKeysUn[idx1]
                                                                       : (idx1 < currFrameNLeft) ? currFrameMvKeys[idx1]
                                                                                                 : currFrameMvKeysRight[idx1 - currFrameNLeft];

        const bool bRight1 = (currFrameNLeft == -1 || idx1 < currFrameNLeft) ? false : true;

        const uint8_t *d1 = &currFrameDescriptors[idx1];

        int bestDist = MATCH_TH_LOW;
        int bestIdx2 = -1;

        size_t neighFeatVecCount = (neighFrameFeatVecCorrespondenceIdx == 0) ? neighFramesfeatVecIdxs[neighborIdx*featVecSize + neighFrameFeatVecCorrespondenceIdx] : 
                                    neighFramesfeatVecIdxs[neighborIdx*featVecSize + neighFrameFeatVecCorrespondenceIdx] - neighFramesfeatVecIdxs[neighborIdx*featVecSize + neighFrameFeatVecCorrespondenceIdx - 1];
                                               
        int neighFeatStartIdx = (neighFrameFeatVecCorrespondenceIdx == 0) ? neighFramesFeatVecStartIdxs[neighborIdx] : 
                                 neighFramesFeatVecStartIdxs[neighborIdx] + neighFramesfeatVecIdxs[neighFrameFeatVecCorrespondenceIdx - 1];

        for (size_t i2 = 0; i2 < neighFeatVecCount; i2++) {

            size_t idx2 = neighFramesfeatVec[neighFeatStartIdx + i2];

            if (neighFramesMapPointExists[neighborIdx*keyFrameMapPointCount + idx2])
                continue;

            const bool bStereo2 = (!neighFramesCamera2[neighborIdx].isAvailable &&  neighFramesMvuRight[neighborIdx*keyFrameMaxFeatureCount + idx2]>=0);

            const uint8_t *d2 = &neighFramesDescriptors[neighborIdx*keyFrameMaxFeatureCount + idx2];

            const int dist = DescriptorDistance(d1, d2);

            if (dist > MATCH_TH_LOW || dist > bestDist)
                continue;

            const TRACKING_DATA_WRAPPER::CudaKeyPoint &kp2 = 
                (neighFramesNLeft[neighborIdx] == -1) ? neighFramesMvKeysUn[neighborIdx*keyFrameMaxFeatureCount + idx2]
                                                      : (idx2 < neighFramesNLeft[neighborIdx]) ? neighFramesMvKeys[neighborIdx*keyFrameMaxFeatureCount + idx2]
                                                                                               : neighFramesMvKeysRight[neighborIdx*keyFrameMaxFeatureCount + idx2 - neighFramesNLeft[neighborIdx]];

            const bool bRight2 = (neighFramesNLeft[neighborIdx] == -1 || idx2 < neighFramesNLeft[neighborIdx]) ? false : true;

            if (!bStereo1 && !bStereo2 && !currFrameCamera2.isAvailable) {
                const float distex = ep[neighborIdx](0) - kp2.ptx;
                const float distey = ep[neighborIdx](1) - kp2.pty;
                if (distex*distex + distey*distey < 100 * pow(scaleFactor, kp2.octave))
                    continue;
            }

            MAPPING_DATA_WRAPPER::CudaCamera camera1, camera2;

            if (currFrameCamera2.isAvailable && neighFramesCamera2[neighborIdx].isAvailable) {
                if (bRight1 && bRight2) {
                    R12 = Rrr[neighborIdx];
                    t12 = trr[neighborIdx];

                    camera1.toK = currFrameCamera2.toK;
                    camera1.mvParameters = currFrameCamera2.mvParameters;
                    camera2.toK = neighFramesCamera2[neighborIdx].toK;
                    camera2.mvParameters = neighFramesCamera2[neighborIdx].mvParameters;
                }
                else if (bRight1 && !bRight2) {
                    R12 = Rrl[neighborIdx];
                    t12 = trl[neighborIdx];

                    camera1.toK = currFrameCamera2.toK;
                    camera1.mvParameters = currFrameCamera2.mvParameters;
                    camera2.toK = neighFramesCamera1[neighborIdx].toK;
                    camera2.mvParameters = neighFramesCamera1[neighborIdx].mvParameters;
                }
                else if (!bRight1 && bRight2) {
                    R12 = Rlr[neighborIdx];
                    t12 = tlr[neighborIdx];

                    camera1.toK = currFrameCamera1.toK;
                    camera1.mvParameters = currFrameCamera1.mvParameters;
                    camera2.toK = neighFramesCamera2[neighborIdx].toK;
                    camera2.mvParameters = neighFramesCamera2[neighborIdx].mvParameters;
                }
                else {
                    R12 = Rll[neighborIdx];
                    t12 = tll[neighborIdx];

                    camera1.toK = currFrameCamera1.toK;
                    camera1.mvParameters = currFrameCamera1.mvParameters;
                    camera2.toK = neighFramesCamera1[neighborIdx].toK;
                    camera2.mvParameters = neighFramesCamera1[neighborIdx].mvParameters;
                }
            }

            bool epipolarConstrainIsMet;
            float kp1mvLevelSigma2 = pow(scaleFactor, kp1.octave) * pow(scaleFactor, kp1.octave);
            float kp2mvLevelSigma2 = pow(scaleFactor, kp2.octave) * pow(scaleFactor, kp2.octave);

            if (currFrameNLeft == -1)
                epipolarConstrainIsMet = pinholeEpipolarConstrain(camera1, camera2, kp1, kp2, R12, t12, kp1mvLevelSigma2, kp2mvLevelSigma2);
            else
                epipolarConstrainIsMet = fisheyeEpipolarConstrain(camera1, camera2, kp1, kp2, R12, t12, kp1mvLevelSigma2, kp2mvLevelSigma2, camPrecision);

            if (bCoarse || epipolarConstrainIsMet) {
                bestIdx2 = idx2;
                bestDist = dist;
            }
        }

        if (bestIdx2 >= 0)
            matchedPairIndexes[neighborIdx*featVecSize + idx1] = bestIdx2;
    }
}

void SearchForTriangulationKernel::launch(ORB_SLAM3::KeyFrame* mpCurrentKeyFrame, std::vector<ORB_SLAM3::KeyFrame*> vpNeighKFs, 
                                          bool mbMonocular, bool mbInertial, bool recentlyLost, bool mbIMU_BA2, 
                                          std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices, std::vector<size_t> &vpNeighKFsIndexes) {

    bool bCoarse = mbInertial && recentlyLost && mbIMU_BA2;
    size_t featVecSize = FEAT_VEC_MAX_SIZE;
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    // cout << vpNeighKFs.size() << std::endl;    
    cout << "GPU current - Frame ID: " << mpCurrentKeyFrame->mnFrameId << ", mbf: " << mpCurrentKeyFrame->mbf << ", mb: " << mpCurrentKeyFrame->mb << endl << endl;
    
    // cout << "Finding good neighbours...\n";
    for (size_t i = 0; i < vpNeighKFs.size(); i++) {
        // cout << "For iteration\n";

        ORB_SLAM3::KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        Eigen::Vector3f vBaseline = Ow2-Ow1;
        const float baseline = vBaseline.norm();

        if (!mbMonocular) {
            // cout << "GPU - Frame ID: " << pKF2->mnFrameId << ", mbf: " << pKF2->mbf << ", baseline: " << baseline << ", mb: " << pKF2->mb << endl;
            if (baseline < pKF2->mb) {
                // cout << "Continue 1\n";
                continue;
            }
        }
        else {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline / medianDepthKF2;

            if (ratioBaselineDepth < 0.01) {
                // cout << "Continue 2\n";
                continue;
            }
        }
        // cout << "1\n";
        vpNeighKFsIndexes.push_back(i);
        cout << "2\n";
    }

    // cout << "Good neighbours found!\n";

    if (vpNeighKFsIndexes.size() == 0)
        return;

    size_t nn = vpNeighKFsIndexes.size();

    size_t currFrameFeatVecIdxCorrespondences[nn*featVecSize] = {-1};
    size_t neighFramesFeatVecIdxCorrespondences[nn*featVecSize];
    size_t correspondeceCount = 0;

    size_t currFrameFeatVec[featVecSize*MAX_FEATURES_IN_WORD];
    size_t currFrameFeatVecIdxs[featVecSize];
    size_t currFrameFeatVecSize = 0, currFrameFeatVecIdxsSize = 0;

    size_t neighFramesfeatVec[featVecSize*MAX_FEATURES_IN_WORD*nn];
    size_t neighFramesfeatVecIdxs[featVecSize*nn];
    size_t neighFramesFeatVecStartIdxs[nn];
    size_t neighFramesfeatVecSize = 0, neighFramesfeatVecIdxsSize = 0;

    Eigen::Matrix3f Rll[nn], Rlr[nn], Rrl[nn], Rrr[nn];
    Eigen::Vector3f tll[nn], tlr[nn], trl[nn], trr[nn];
    Eigen::Vector2f eps[nn];

    copyFrameFeatVec(mpCurrentKeyFrame, currFrameFeatVec, currFrameFeatVecIdxs, &currFrameFeatVecSize, &currFrameFeatVecIdxsSize);

    cout << "Finding correspondences...\n";

    for (size_t i = 0; i < nn; i++) {

        ORB_SLAM3::KeyFrame* pKF2 = vpNeighKFs[vpNeighKFsIndexes[i]];

        // TODO: should change grid size if continued here
        neighFramesFeatVecStartIdxs[i] = neighFramesfeatVecSize;
        copyFrameFeatVec(pKF2, neighFramesfeatVec + neighFramesfeatVecSize, neighFramesfeatVecIdxs + neighFramesfeatVecIdxsSize,
                         &neighFramesfeatVecSize, &neighFramesfeatVecIdxsSize);

        const DBoW2::FeatureVector &vFeatVec1 = mpCurrentKeyFrame->mFeatVec;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        //Compute epipole in second image
        Sophus::SE3f T1w = mpCurrentKeyFrame->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();
        Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
        Eigen::Vector3f Cw = mpCurrentKeyFrame->GetCameraCenter();
        Eigen::Vector3f C2 = T2w * Cw;

        Eigen::Vector2f ep = pKF2->mpCamera->project(C2);
        Sophus::SE3f T12;
        Sophus::SE3f Tll, Tlr, Trl, Trr;
        Eigen::Matrix3f R12; // for fastest computation
        Eigen::Vector3f t12; // for fastest computation

        eps[i] = ep;

        ORB_SLAM3::GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        if (!mpCurrentKeyFrame->mpCamera2 && !pKF2->mpCamera2){
            T12 = T1w * Tw2;
            R12 = T12.rotationMatrix();
            t12 = T12.translation();
        }
        else {
            Sophus::SE3f Tr1w = mpCurrentKeyFrame->GetRightPose();
            Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
            Tll = T1w * Tw2;
            Tlr = T1w * Twr2;
            Trl = Tr1w * Tw2;
            Trr = Tr1w * Twr2;
        }

        Rll[i] = Tll.rotationMatrix();
        Rlr[i] = Tlr.rotationMatrix();
        Rrl[i] = Trl.rotationMatrix();
        Rrr[i] = Trr.rotationMatrix();
        tll[i] = Tll.translation();
        tlr[i] = Tlr.translation();
        trl[i] = Trl.translation();
        trr[i] = Trr.translation();

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
        size_t f1Idx = 0, f2Idx = 0;
        int counter = 0;

        while (f1it != f1end && f2it != f2end) {
            if (f1it->first == f2it->first) {
                currFrameFeatVecIdxCorrespondences[i*featVecSize + counter] = f1Idx;
                neighFramesFeatVecIdxCorrespondences[i*featVecSize + counter] = f2Idx;

                f1it++;
                f2it++;
                f1Idx++;
                f2Idx++;
                counter++;
            }
            else if (f1it->first < f2it->first) {
                f1it = vFeatVec1.lower_bound(f2it->first);
                f1Idx = std::distance(vFeatVec1.begin(), f1it);
            }
            else {
                f2it = vFeatVec2.lower_bound(f1it->first);
                f2Idx = std::distance(vFeatVec2.begin(), f2it);
            }
        }

    }

    cout << "Correspondences found!\n";

    int maxFeatures = CudaUtils::nFeatures_with_th;
    size_t mapPointVecSize;
    
    if (CudaUtils::cameraIsFisheye)
        mapPointVecSize = maxFeatures*2;
    else 
        mapPointVecSize = maxFeatures;

    cout << "Creating Nleft!\n";

    size_t neighFramesNLeft[nn];
    for (int i = 0; i < nn; i++) 
        neighFramesNLeft[i] = vpNeighKFs[i]->NLeft;

    cout << "Creating mvuRight!\n";

    float neighFramesMvuRight[maxFeatures*nn];
    for (int i = 0; i < nn; i++)
        memcpy(neighFramesMvuRight + i*maxFeatures, vpNeighKFs[i]->mvuRight.data(), sizeof(float)*vpNeighKFs[i]->mvuRight.size());

    bool currFrameMapPointExists[mapPointVecSize], neighFramesMapPointExists[mapPointVecSize*nn];
    uchar *neighFramesDescriptors[maxFeatures*nn];

    TRACKING_DATA_WRAPPER::CudaKeyPoint currFrameMvKeysUn[maxFeatures], neighFramesMvKeysUn[maxFeatures*nn];
    TRACKING_DATA_WRAPPER::CudaKeyPoint currFrameMvKeys[maxFeatures], neighFramesMvKeys[maxFeatures*nn];
    TRACKING_DATA_WRAPPER::CudaKeyPoint currFrameMvKeysRight[maxFeatures], neighFramesMvKeysRight[maxFeatures*nn];

    cout << "Creating mapPoints!\n";

    mpCurrentKeyFrame->GetMapPointAvailabality(currFrameMapPointExists);
    for (int i = 0; i < nn; i++)
        vpNeighKFs[i]->GetMapPointAvailabality(neighFramesMapPointExists + i*mapPointVecSize);

    cout << "Creating descripors!\n";

    for (size_t i = 0; i < nn; i++)
        memcpy(neighFramesDescriptors + i*maxFeatures, vpNeighKFs[i]->mDescriptors.data, sizeof(uchar)*maxFeatures);

    cout << "Creating keypoints!\n";

    if (CudaUtils::cameraIsFisheye) {
        copyGPUKeypoints(currFrameMvKeysUn, mpCurrentKeyFrame->mvKeysUn);
        for (int i = 0; i < nn; i++)
            copyGPUKeypoints(neighFramesMvKeysUn + i*maxFeatures, vpNeighKFs[i]->mvKeysUn);
    }
    else {
        copyGPUKeypoints(currFrameMvKeys, mpCurrentKeyFrame->mvKeys);
        for (int i = 0; i < nn; i++)
            copyGPUKeypoints(neighFramesMvKeys + i*maxFeatures, vpNeighKFs[i]->mvKeys);

        copyGPUKeypoints(currFrameMvKeysRight, mpCurrentKeyFrame->mvKeysRight);
        for (int i = 0; i < nn; i++)
            copyGPUKeypoints(neighFramesMvKeysRight + i*maxFeatures, vpNeighKFs[i]->mvKeysRight);
    }

    cout << "Creating camera!\n";

    MAPPING_DATA_WRAPPER::CudaCamera currFrameCamera1, currFrameCamera2, neighFramesCamera1[nn], neighFramesCamera2[nn];

    copyGPUCamera(&currFrameCamera1, mpCurrentKeyFrame->mpCamera);
    copyGPUCamera(&currFrameCamera2, mpCurrentKeyFrame->mpCamera2);
    for (int i = 0; i < nn; i++)
        copyGPUCamera(&neighFramesCamera1[i], vpNeighKFs[i]->mpCamera);
    for (int i = 0; i < nn; i++)
        copyGPUCamera(&neighFramesCamera2[i], vpNeighKFs[i]->mpCamera2);

    cout << "Data trasnfer to gpu started...\n";

    // Transfer data to GPU
    checkCudaError(cudaMemcpy(d_currFrameFeatVec, currFrameFeatVec, sizeof(size_t)*featVecSize*MAX_FEATURES_IN_WORD, cudaMemcpyHostToDevice), "Failed to copy vector currFrameFeatVec from host to device");
    checkCudaError(cudaMemcpy(d_currFrameFeatVecIdxs, currFrameFeatVecIdxs, sizeof(size_t)*featVecSize, cudaMemcpyHostToDevice), "Failed to copy vector currFrameFeatVecIdxs from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesfeatVec, neighFramesfeatVec, sizeof(size_t)*featVecSize*MAX_FEATURES_IN_WORD*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesfeatVec from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesfeatVecIdxs, neighFramesfeatVecIdxs, sizeof(size_t)*featVecSize*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesfeatVecIdxs from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesFeatVecStartIdxs, neighFramesFeatVecStartIdxs, sizeof(size_t)*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesFeatVecStartIdxs from host to device");

    checkCudaError(cudaMemcpy(d_currFrameFeatVecIdxCorrespondences, currFrameFeatVecIdxCorrespondences, sizeof(size_t)*nn*featVecSize, cudaMemcpyHostToDevice), "Failed to copy vector currFrameFeatVecIdxCorrespondences from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesFeatVecIdxCorrespondences, neighFramesFeatVecIdxCorrespondences, sizeof(size_t)*nn*featVecSize, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesFeatVecIdxCorrespondences from host to device");

    checkCudaError(cudaMemcpy(d_neighFramesNLeft, neighFramesNLeft, sizeof(size_t)*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesNLeft from host to device");

    checkCudaError(cudaMemcpy(d_Rll, Rll, sizeof(Eigen::Matrix3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector Rll from host to device");
    checkCudaError(cudaMemcpy(d_Rlr, Rlr, sizeof(Eigen::Matrix3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector Rlr from host to device");
    checkCudaError(cudaMemcpy(d_Rrl, Rrl, sizeof(Eigen::Matrix3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector Rrl from host to device");
    checkCudaError(cudaMemcpy(d_Rrr, Rrr, sizeof(Eigen::Matrix3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector Rrr from host to device");
    checkCudaError(cudaMemcpy(d_tll, tll, sizeof(Eigen::Vector3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector tll from host to device");
    checkCudaError(cudaMemcpy(d_tlr, tlr, sizeof(Eigen::Vector3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector tlr from host to device");
    checkCudaError(cudaMemcpy(d_trl, trl, sizeof(Eigen::Vector3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector trl from host to device");
    checkCudaError(cudaMemcpy(d_trr, trr, sizeof(Eigen::Vector3f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector trr from host to device");

    checkCudaError(cudaMemcpy(d_currFrameMapPointExists, currFrameMapPointExists, sizeof(currFrameMapPointExists), cudaMemcpyHostToDevice), "Failed to copy vector currFrameMapPointExists from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesMapPointExists, neighFramesMapPointExists, sizeof(neighFramesMapPointExists), cudaMemcpyHostToDevice), "Failed to copy vector neighFramesMapPointExists from host to device");

    checkCudaError(cudaMemcpy(d_currFrameDescriptors, mpCurrentKeyFrame->mDescriptors.data, sizeof(uchar)*maxFeatures, cudaMemcpyHostToDevice), "Failed to copy vector currFrameDescriptors from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesDescriptors, neighFramesDescriptors, sizeof(uchar)*maxFeatures*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesDescriptors from host to device");
    
    checkCudaError(cudaMemcpy(d_currFrameMvuRight, mpCurrentKeyFrame->mvuRight.data(), sizeof(float)*mpCurrentKeyFrame->mvuRight.size(), cudaMemcpyHostToDevice), "Failed to copy vector currFrameMvuRight from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesMvuRight, neighFramesMvuRight, sizeof(float)*maxFeatures*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesMvuRight from host to device");

    checkCudaError(cudaMemcpy(d_neighFramesCamera1, neighFramesCamera1, sizeof(MAPPING_DATA_WRAPPER::CudaCamera)*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesCamera1 from host to device");
    checkCudaError(cudaMemcpy(d_neighFramesCamera2, neighFramesCamera2, sizeof(MAPPING_DATA_WRAPPER::CudaCamera)*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesCamera2 from host to device");

    checkCudaError(cudaMemcpy(d_ep, eps, sizeof(Eigen::Vector2f)*nn, cudaMemcpyHostToDevice), "Failed to copy vector eps from host to device");

    if (CudaUtils::cameraIsFisheye) {
        checkCudaError(cudaMemcpy(d_currFrameMvKeysUn, currFrameMvKeysUn, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures, cudaMemcpyHostToDevice), "Failed to copy vector currFrameMvKeysUn from host to device");
        checkCudaError(cudaMemcpy(d_neighFramesMvKeysUn, neighFramesMvKeysUn, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures*nn, cudaMemcpyHostToDevice), "Failed to copy vector currFrameMvKeysUn from host to device");
    }
    else {
        checkCudaError(cudaMemcpy(d_currFrameMvKeys, currFrameMvKeys, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures, cudaMemcpyHostToDevice), "Failed to copy vector currFrameMvKeys from host to device");
        checkCudaError(cudaMemcpy(d_neighFramesMvKeys, neighFramesMvKeys, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesMvKeys from host to device");
        checkCudaError(cudaMemcpy(d_currFrameMvKeysRight, currFrameMvKeysRight, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures, cudaMemcpyHostToDevice), "Failed to copy vector currFrameMvKeysRight from host to device");
        checkCudaError(cudaMemcpy(d_neighFramesMvKeysRight, neighFramesMvKeysRight, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures*nn, cudaMemcpyHostToDevice), "Failed to copy vector neighFramesMvKeysRight from host to device");
    }

    cout << "GPU data transfer finished!\n";

    float camPrecision;

    if (CudaUtils::cameraIsFisheye) {
        ORB_SLAM3::KannalaBrandt8* pKBCam = (ORB_SLAM3::KannalaBrandt8*) mpCurrentKeyFrame->mpCamera;
        camPrecision = pKBCam->GetPrecision();
    }

    cout << "Entering kernel...\n";

    // Run the Kernel
    dim3 gridDim(nn);
    dim3 blockDim(featVecSize);
    searchForTriangulationKernel<<<gridDim, blockDim>>>(
        featVecSize, mapPointVecSize, maxFeatures, CudaUtils::scaleFactor, camPrecision, bCoarse,
        d_currFrameFeatVecIdxCorrespondences, d_neighFramesFeatVecIdxCorrespondences,
        d_currFrameFeatVec, d_currFrameFeatVecIdxs, d_neighFramesfeatVec, d_neighFramesfeatVecIdxs, d_neighFramesFeatVecStartIdxs,
        mpCurrentKeyFrame->NLeft, d_neighFramesNLeft,d_Rll, d_Rlr, d_Rrl, d_Rrr, d_tll, d_tlr, d_trl, d_trr, 
        d_currFrameMapPointExists, d_neighFramesMapPointExists, d_currFrameDescriptors, d_neighFramesDescriptors, 
        d_currFrameMvKeysUn, d_neighFramesMvKeysUn, d_currFrameMvKeys, d_neighFramesMvKeys, d_currFrameMvKeysRight, d_neighFramesMvKeysRight,
        d_currFrameMvuRight, d_neighFramesMvuRight,d_currFrameCamera1, d_currFrameCamera2, d_neighFramesCamera1, d_neighFramesCamera2, d_ep,
        d_matchedPairIndexes
    );

    checkCudaError(cudaGetLastError(), "Failed to launch searchForTriangulationKernel kernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code after launching the kernel");

    cout << "Kernel finished!\n";
    
    size_t h_matchedPairIndexes[featVecSize*nn];
    checkCudaError(cudaMemcpy(h_matchedPairIndexes, d_matchedPairIndexes, sizeof(size_t)*featVecSize*nn, cudaMemcpyDeviceToHost), "Failed to copy vector d_matchedPairIndexes from device to host");

    convertToVectorOfPairs(h_matchedPairIndexes, vpNeighKFsIndexes, featVecSize, allvMatchedIndices);
}

void SearchForTriangulationKernel::copyGPUKeypoints(TRACKING_DATA_WRAPPER::CudaKeyPoint* out, const std::vector<cv::KeyPoint> keypoints) {
    for (int i = 0; i < keypoints.size(); i++) {
        out[i].ptx = keypoints[i].pt.x;
        out[i].pty = keypoints[i].pt.y;
        out[i].octave = keypoints[i].octave;
    }
}

void SearchForTriangulationKernel::copyGPUCamera(MAPPING_DATA_WRAPPER::CudaCamera *out, ORB_SLAM3::GeometricCamera *camera) {
    out->isAvailable = (bool) camera;
    memcpy(out->mvParameters, camera->getParameters().data(), sizeof(float)*camera->getParameters().size());
    out->toK = camera->toK_();
}

void SearchForTriangulationKernel::copyFrameFeatVec(ORB_SLAM3::KeyFrame* kf, size_t* outFeatureVec, size_t* outFeatureVecIdxs, 
                                                    size_t* outFeatureVecLastIdx, size_t* outFeatureVecIdxsLastIdx) {
                                            
    DBoW2::FeatureVector::const_iterator frameIt = kf->mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator frameEnd = kf->mFeatVec.end();
    size_t outFeatureVecSize = 0, counter = 0;

    while (frameIt != frameEnd) {
        outFeatureVecSize += frameIt->second.size();

        memcpy(outFeatureVec + outFeatureVecSize, frameIt->second.data(), sizeof(size_t) * frameIt->second.size());
        outFeatureVecIdxs[counter] = outFeatureVecSize;
        
        counter++;
        frameIt++;
    }

    *outFeatureVecLastIdx += outFeatureVecSize;
    *outFeatureVecIdxsLastIdx += counter;
}

std::vector<std::vector<std::pair<size_t,size_t>>> SearchForTriangulationKernel::convertToVectorOfPairs(
    size_t* gpuMatchedIdxs, std::vector<size_t> vpNeighKFsIndexes, int featVecSize, std::vector<std::vector<std::pair<size_t,size_t>>> &allvMatchedIndices
) {
    for (size_t i = 0; i < vpNeighKFsIndexes.size(); i++) {
        std::vector<std::pair<size_t,size_t>> tmp;
        for (size_t j = 0; j < featVecSize; j++) {
            if (gpuMatchedIdxs[i*featVecSize + j] != -1)
                tmp.push_back(std::make_pair(j, gpuMatchedIdxs[i*featVecSize + j]));
        }
        allvMatchedIndices.push_back(tmp);
    }
}

void SearchForTriangulationKernel::initialize() {
    
    if (memory_is_initialized)
        return;
    
    cout << "GPU memory init started...\n";

    size_t featVecSize = FEAT_VEC_MAX_SIZE;
    size_t maxNeighborCount = MAX_NEIGHBOUR_COUNT;
    int maxFeatures = CudaUtils::nFeatures_with_th;
    size_t mapPointVecSize;
    if (CudaUtils::cameraIsFisheye)
        mapPointVecSize = maxFeatures*2;
    else 
        mapPointVecSize = maxFeatures;


    checkCudaError(cudaMalloc((void**)&d_currFrameFeatVec, sizeof(size_t)*featVecSize*MAX_FEATURES_IN_WORD), "Failed to allocate device vector d_currFrameFeatVec");
    checkCudaError(cudaMalloc((void**)&d_currFrameFeatVecIdxs, sizeof(size_t)*featVecSize), "Failed to allocate device vector d_currFrameFeatVecIdxs");
    checkCudaError(cudaMalloc((void**)&d_neighFramesfeatVec, sizeof(size_t)*featVecSize*MAX_FEATURES_IN_WORD*maxNeighborCount), "Failed to allocate device vector d_neighFramesfeatVec");
    checkCudaError(cudaMalloc((void**)&d_neighFramesfeatVecIdxs, sizeof(size_t)*featVecSize*maxNeighborCount), "Failed to allocate device vector d_neighFramesfeatVecIdxs");
    checkCudaError(cudaMalloc((void**)&d_neighFramesFeatVecStartIdxs, sizeof(size_t)*maxNeighborCount), "Failed to allocate device vector d_neighFramesFeatVecStartIdxs");
    checkCudaError(cudaMalloc((void**)&d_currFrameFeatVecIdxCorrespondences, sizeof(size_t)*featVecSize*maxNeighborCount), "Failed to allocate device vector d_currFrameFeatVecIdxCorrespondences");
    checkCudaError(cudaMalloc((void**)&d_neighFramesFeatVecIdxCorrespondences, sizeof(size_t)*featVecSize*maxNeighborCount), "Failed to allocate device vector d_neighFramesFeatVecIdxCorrespondences");
    checkCudaError(cudaMalloc((void**)&d_currFrameMapPointExists, sizeof(bool)*mapPointVecSize), "Failed to allocate device vector d_currFrameMapPointExists");
    checkCudaError(cudaMalloc((void**)&d_neighFramesMapPointExists, sizeof(bool)*mapPointVecSize*maxNeighborCount), "Failed to allocate device vector d_neighFramesMapPointExists");
    checkCudaError(cudaMalloc((void**)&d_currFrameDescriptors, sizeof(uchar)*maxFeatures*DESCRIPTOR_SIZE), "Failed to allocate device vector d_currFrameDescriptors");
    checkCudaError(cudaMalloc((void**)&d_neighFramesDescriptors, sizeof(uchar)*maxFeatures*DESCRIPTOR_SIZE*maxNeighborCount), "Failed to allocate device vector d_neighFramesDescriptors");
    checkCudaError(cudaMalloc((void**)&d_currFrameMvuRight, sizeof(float)*maxFeatures), "Failed to allocate device vector d_currFrameMvuRight");
    checkCudaError(cudaMalloc((void**)&d_neighFramesMvuRight, sizeof(float)*maxFeatures*maxNeighborCount), "Failed to allocate device vector d_neighFramesMvuRight");
    checkCudaError(cudaMalloc((void**)&d_neighFramesCamera1, sizeof(MAPPING_DATA_WRAPPER::CudaCamera)*maxNeighborCount), "Failed to allocate device vector d_neighFramesCamera1");
    checkCudaError(cudaMalloc((void**)&d_neighFramesCamera2, sizeof(MAPPING_DATA_WRAPPER::CudaCamera)*maxNeighborCount), "Failed to allocate device vector d_neighFramesCamera2");
    checkCudaError(cudaMalloc((void**)&d_ep, sizeof(Eigen::Vector2f)*maxNeighborCount), "Failed to allocate device vector d_ep");

    if (CudaUtils::cameraIsFisheye) {
        checkCudaError(cudaMalloc((void**)&d_currFrameMvKeysUn, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures), "Failed to allocate device vector d_currFrameMvKeysUn");
        checkCudaError(cudaMalloc((void**)&d_neighFramesMvKeysUn, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures*maxNeighborCount), "Failed to allocate device vector d_neighFramesMvKeysUn");
    }
    else {
        checkCudaError(cudaMalloc((void**)&d_currFrameMvKeys, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures), "Failed to allocate device vector d_currFrameMvKeys");
        checkCudaError(cudaMalloc((void**)&d_neighFramesMvKeys, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures*maxNeighborCount), "Failed to allocate device vector d_neighFramesMvKeys");
        checkCudaError(cudaMalloc((void**)&d_currFrameMvKeysRight, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures), "Failed to allocate device vector d_currFrameMvKeysRight");
        checkCudaError(cudaMalloc((void**)&d_neighFramesMvKeysRight, sizeof(TRACKING_DATA_WRAPPER::CudaKeyPoint)*maxFeatures*maxNeighborCount), "Failed to allocate device vector d_neighFramesMvKeysRight");
    }
    
    checkCudaError(cudaMalloc((void**)&d_matchedPairIndexes, sizeof(size_t)*featVecSize*maxNeighborCount), "Failed to allocate device vector d_matchedPairIndexes");

    memory_is_initialized = true;

    cout << "GPU memory init ended!\n";
}

void SearchForTriangulationKernel::shutdown() {
    if (memory_is_initialized) {
        cudaFree(d_currFrameFeatVec);
        cudaFree(d_currFrameFeatVecIdxs);
        cudaFree(d_neighFramesfeatVec);
        cudaFree(d_neighFramesfeatVecIdxs);        
        cudaFree(d_neighFramesFeatVecStartIdxs);
        cudaFree(d_currFrameFeatVecIdxCorrespondences);
        cudaFree(d_neighFramesFeatVecIdxCorrespondences);
        cudaFree(d_currFrameDescriptors);
        cudaFree(d_neighFramesDescriptors);
        cudaFree(d_currFrameMvuRight);
        cudaFree(d_neighFramesMvuRight);
        cudaFree(d_neighFramesCamera1);
        cudaFree(d_neighFramesCamera2);
        cudaFree(d_ep);

        cudaFree(d_matchedPairIndexes);

        if (CudaUtils::cameraIsFisheye) {
            cudaFree(d_currFrameMvKeysUn);
            cudaFree(d_neighFramesMvKeysUn);
        }
        else {
            cudaFree(d_currFrameMvKeys);
            cudaFree(d_neighFramesMvKeys);
            cudaFree(d_currFrameMvKeysRight);
            cudaFree(d_neighFramesMvKeysRight);
        }
    }
}

void SearchForTriangulationKernel::saveStats(const std::string &file_path) {

    // std::string data_path = file_path + "/SearchForTriangulationKernel/";
    // std::cout << "[SearchForTriangulationKernel:] writing stats data into file: " << data_path << '\n';
    // if (mkdir(data_path.c_str(), 0755) == -1) {
    //     std::cerr << "[SearchForTriangulationKernel:] Error creating directory: " << strerror(errno) << std::endl;
    // }
    // std::ofstream myfile;
    
    // myfile.open(data_path + "/total_exec_time.txt");
    // for (const auto& p : total_exec_time) {
    //     myfile << p.first << ": " << p.second << std::endl;
    // }
    // myfile.close();

    // myfile.open(data_path + "/kernel_exec_time.txt");
    // for (const auto& p : kernel_exec_time) {
    //     myfile << p.first << ": " << p.second << std::endl;
    // }
    // myfile.close();

    // myfile.open(data_path + "/data_transfer_time.txt");
    // for (const auto& p : data_transfer_time) {
    //     myfile << p.first << ": " << p.second << std::endl;
    // }
    // myfile.close();

    // myfile.open(data_path + "/input_data_transfer_time.txt");
    // for (const auto& p : input_data_transfer_time) {
    //     myfile << p.first << ": " << p.second << std::endl;
    // }
    // myfile.close();

    // myfile.open(data_path + "/output_data_transfer_time.txt");
    // for (const auto& p : output_data_transfer_time) {
    //     myfile << p.first << ": " << p.second << std::endl;
    // }
    // myfile.close();

    // myfile.open(data_path + "/data_wrap_time.txt");
    // for (const auto& p : data_wrap_time) {
    //     myfile << p.first << ": " << p.second << std::endl;
    // }
    // myfile.close();

    // myfile.open(data_path + "/input_data_wrap_time.txt");
    // for (const auto& p : input_data_wrap_time) {
    //     myfile << p.first << ": " << p.second << std::endl;
    // }
    // myfile.close();

    // myfile.open(data_path + "/output_data_wrap_time.txt");
    // for (const auto& p : output_data_wrap_time) {
    //     myfile << p.first << ": " << p.second << std::endl;
    // }
    // myfile.close();
}