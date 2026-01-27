#include "Stats/LoopClosingStats.h"
#include <sstream>  

using namespace std;

void LoopClosingStats::saveStats(const string &file_path) {
#ifdef REGISTER_LOOP_CLOSING_STATS
    string data_path = file_path + "/LoopClosing";
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[LoopClosingStats:] Error creating directory: " << strerror(errno) << std::endl;
    }

    data_path = data_path + "/data/";
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[LoopClosingStats:] Error creating directory: " << strerror(errno) << std::endl;
    }
    cout << "Writing stats data into file: " << data_path << '\n';

    std::ofstream myfile;

    myfile.open(data_path + "/loop_closing_time.txt");
    for (size_t i = 0; i < loopClosing_time.size(); ++i) {
        myfile << i << ": " << loopClosing_time[i] << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/optimizeEssentialGraph_time.txt");
    for (size_t i = 0; i < optimizeEssentialGraph_time.size(); ++i) {
        myfile << i << ": " << optimizeEssentialGraph_time[i] << std::endl;
    }
    myfile.close();
    
#endif
}