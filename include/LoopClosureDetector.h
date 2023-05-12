#ifndef LOOPCLOSUREDETECTOR_H
#define LOOPCLOSUREDETECTOR_H

class LoopClosureDetector {
public:
    static LoopClosureDetector& instance();

    bool isLoopClosureDetected() const;
    void setLoopClosureDetected(bool val);

private:
    LoopClosureDetector();
    bool loopClosureDetected;
};

#endif // LOOPCLOSUREDETECTOR_H