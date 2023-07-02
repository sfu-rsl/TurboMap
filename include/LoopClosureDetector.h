#ifndef LOOPCLOSUREDETECTOR_H
#define LOOPCLOSUREDETECTOR_H

class LoopClosureDetector {
public:
    static LoopClosureDetector& instance();

    bool isLoopClosureDetected() const;
    void setLoopClosureDetected(bool val);

    bool isMergeDetected() const;
    void setMergeDetected(bool val);

private:
    LoopClosureDetector();
    bool loopClosureDetected;
    bool mergeDetected;
};

#endif // LOOPCLOSUREDETECTOR_H