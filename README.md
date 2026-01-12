# TurboMap

**TurboMap** is a **GPU-accelerated and CPU-optimized local mapping module** for **ORB-SLAM3**. It speeds up the most time-consuming parts of local mapping by offloading key operations to the GPU and optimizing the rest on the CPU.

TurboMap accelerates the following components:
- 🔹 **Search for Triangulation** — implemented with custom CUDA kernels  
- 🔹 **Map-Point Fusion** — GPU-accelerated with CUDA  
- 🔹 **Local Keyframe Culling** — optimized on CPU  
- 🔹 **Local Bundle Adjustment** — uses an existing GPU-accelerated backend ([compute-engine](https://github.com/sfu-rsl/compute-engine) and [gpu-block-solver](https://github.com/sfu-rsl/gpu-block-solver))

In our tests, **TurboMap** maintained the same accuracy as the original **ORB-SLAM3** system while achieving the following performance gains:
- ⚡ **1.3× speedup** on the **EuRoC dataset**  
- ⚡ **1.6× speedup** on the **TUM-VI dataset**

We evaluated TurboMap using **stereo-inertial configurations** on both **desktop** and **embedded platforms**. The machine specifications used for testing are as follows:

<div align="center">
<table>
  <thead>
    <tr>
      <th>Machine</th>
      <th>Specs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Desktop</td>
      <td>
        20-core Intel Core i7-12700K CPU @ 5.0 GHz <br>
        NVIDIA RTX 3090 GPU (10496-core) <br>
        64 GB RAM
      </td>
    </tr>
    <tr>
      <td>Xavier NX</td>
      <td>
        6-core ARM Carmel CPU @ 1.4 GHz <br>
        NVIDIA Volta GPU (384-core) <br>
        8 GB RAM
      </td>
    </tr>
  </tbody>
</table>

<strong>Table 1: Machine Specifications</strong>
</div>

<br>

The following diagram shows the data flow of the local mapping process in TurboMap, with CPU-side components at the top, GPU-side components at the bottom, arrows indicating data transfers between modules, and a persistent keyframe storage maintained on the GPU:

<p align="center">
  <img src="https://github.com/sfu-rsl/TurboMap/blob/main/TurboMap_Local_Mapping_Overview.png" alt="Tracking In FastTrack">
</p>
<p align="center"><strong>Figure 1: Local Mapping Overview in TurboMap</strong></p>

<br>

⏳ TurboMap has been submitted for publication at [ICRA 2026](https://2026.ieee-icra.org/). You can check out the paper [here](https://arxiv.org/abs/2511.02036).

🚀 You can also check out our related project, [FastTrack](https://github.com/sfu-rsl/FastTrack), which this work builds upon and which has been accepted for publication at IROS 2025.

# 2. Prerequisites
We have tested the library in **Ubuntu 22.04** and **20.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++14 Compiler
You need C++14 installed to build and run TurboMap.

## Cuda
We have tested the library with **Cuda 12.5**. Download and install instructions can be found at: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 4.4.0**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Download and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

Pangolin avoids to use Eigen in CUDA. To compile this project, guards in line 475 of glsl.hpp and 47 of glsl.h should be commented. Checkout [this issue](https://github.com/stevenlovegrove/Pangolin/issues/814) in ORB-SLAM3 github issues. 

## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## Python
Required to calculate the alignment of the trajectory with the ground truth. **Required Numpy module**.

* (win) http://www.python.org/downloads/windows
* (deb) `sudo apt install libpython2.7-dev`
* (mac) preinstalled with osx

## Compute-engine Dependencies
TurboMap relies on an external library to accelerate Local Bundle Adjustment (LBA). Make sure to install its dependencies, listed [here](https://github.com/sfu-rsl/compute-engine#building).

# 3. Building TurboMap 

Clone the repository:
```
git clone git@github.com:sfu-rsl/FastTrack.git
```

After cloning, follow the instructions [here](https://github.com/sfu-rsl/gpu-block-solver/blob/master/INTEGRATION.md) to download and install the accelerated GPU block solver. Note that some of the instructions have already been applied (like the changes in cmake), so you don't have to do them again.

Our system is based on ORB-SLAM3 and ORB-SLAM3 provides a script `build.sh` to build the *Thirdparty* libraries and *ORB-SLAM3*. Please make sure you have installed all required dependencies (see section 2). Execute:
```
cd TurboMap
chmod +x build.sh
./build.sh
```

This will create **libORB_SLAM3.so**  at *lib* folder and the executables in *Examples* folder.

# 4. Running TurboMap
## EuRoC Examples
[EuRoC dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) was recorded with two pinhole cameras and an inertial sensor. We provide an example script to launch EuRoC sequences in all the sensor configurations.

1. Download a sequence (ASL format) from http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

2. Open the script "euroc_eval_examples.sh" in the **Examples** directory of the project. Change **pathDatasetEuroc** variable to point to the directory where the dataset has been uncompressed. 

3. Execute the following script to run a single sequence:
```
./run_script.sh <dataset_name> <[0] for ORB-SLAM3, [1] for FastTrack, [2] for TurboMap> <[0] for STDOUT output, [1] for file output> <run_version> <FastMap_mode>
```
Using **`FastMap_mode`**, you can enable or disable each TurboMap kernel individually. It consists of **four digits**, each representing whether a specific task is optimized:

1. **First digit:** Search for Triangulation on GPU  
2. **Second digit:** Map-Point Fusion on GPU  
3. **Third digit:** Optimized Keyframe Culling on CPU  
4. **Fourth digit:** GPU-accelerated Local Bundle Adjustment (LBA)  

For example, setting `FastMap_mode = 1111` runs all accelerations. If you don't want to run TurboMap, simply omit this argument.

4. Alternatively, you can use this command to run multiple sequences for multiple times:
```
./run_experiments.sh <[0] for ORB-SLAM3, [1] for FastTrack, [2] for TurboMap> <version> <num_iterations>
```

## TUM-VI Examples
[TUM-VI dataset](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) was recorded with two fisheye cameras and an inertial sensor.

1. Download a sequence from https://vision.in.tum.de/data/datasets/visual-inertial-dataset and uncompress it.

2. Open the script "tum_vi_examples.sh" in the root of the project. Change **pathDatasetTUM_VI** variable to point to the directory where the dataset has been uncompressed. 

3. Execute it like the EuRoC dataset.
