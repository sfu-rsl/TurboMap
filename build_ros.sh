echo "Building ROS nodes"

<<<<<<< HEAD
cd Examples/ROS/ORB_SLAM3
=======
cd Examples_old/ROS/ORB_SLAM3
>>>>>>> ORB_SLAM3_pg-opt/pg-opt
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j
