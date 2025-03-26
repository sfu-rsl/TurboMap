#ifndef LOCAL_MAPPING_STATS_H
#define LOCAL_MAPPING_STATS_H

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include "Stats/StatsInterface.h"
#include "Kernels/CudaUtils.h"
#include "Kernels/MappingKernelController.h"

using namespace std;

class LocalMappingStats: public StatsInterface {
    public:
        static LocalMappingStats& getInstance() {
            static LocalMappingStats instance;
            return instance;
        }
        void saveStats(const string &file_path) override;

    public:
        std::vector<double> localMapping_time;
        std::vector<double> processKF_time;
        std::vector<double> MPCulling_time;
        std::vector<double> MPCreation_time;
        std::vector<double> searchInNeighbors_time;
        std::vector<double> LBA_time;
        std::vector<double> KFCulling_time;
        std::vector<double> searchForTriangulation_time;
        std::vector<double> fuse_time;
        std::vector<int> createdMappoints_num;
        double searchForTriangulation_init_time;

        std::vector<double> eraseCudaKeyFrameMapPoint_time;
        std::vector<double> updateCudaKeyFrameMapPoint_time;
        std::vector<double> addCudaKeyFrame_time;
        std::vector<double> eraseCudaKeyFrame_time;
        std::vector<double> addFeatureVector_time;

        std::vector<double> replaceCudaMapPoint_time;
        std::vector<double> updateCudaMapPointObservations_time;
        std::vector<double> updateCudaMapPointWorldPos_time;
        std::vector<double> updateCudaMapNormalAndDepth_time;
        std::vector<double> updateCudaMapPointDescriptor_time;
        std::vector<double> addCudaMapPoint_time;
        std::vector<double> eraseCudaMapPoint_time;

    private:
        LocalMappingStats() = default; // Private constructor
};

#endif 