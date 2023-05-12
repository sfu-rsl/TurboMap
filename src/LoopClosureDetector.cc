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

LoopClosureDetector::LoopClosureDetector() : loopClosureDetected(false) {}
