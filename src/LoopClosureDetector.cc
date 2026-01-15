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

LoopClosureDetector::LoopClosureDetector() : loopClosureDetected(false), mergeDetected(false) {}
