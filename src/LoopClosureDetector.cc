#include "LoopClosureDetector.h"

LoopClosureDetector& LoopClosureDetector::instance() {
    static LoopClosureDetector instance;
    return instance;
}

bool LoopClosureDetector::isLoopClosureDetected() const {
    return loopClosureDetected;
}

void LoopClosureDetector::setLoopClosureDetected(bool val) {
    loopClosureDetected = val;
}

bool LoopClosureDetector::isMergeDetected() const {
    return mergeDetected;
}

void LoopClosureDetector::setMergeDetected(bool val) {
    mergeDetected = val;
}

bool LoopClosureDetector::getJacobiGPUStatus(void) {
    return JacobiGPUStatus;
}

void LoopClosureDetector::setJacobiGPUStatus(bool status) {
    JacobiGPUStatus = status;
}

LoopClosureDetector::LoopClosureDetector() : loopClosureDetected(false), mergeDetected(false) {}
