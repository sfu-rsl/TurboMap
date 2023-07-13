echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release -DG2O_USE_OPENMP=ON
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../pose-graph-optimizer

echo "Configuring and building Thirdparty/pose-graph-optimizer ..."
mkdir build
cd build
cmake ..
make -j

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building ORB_SLAM3 ..."

mkdir build
cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release -DG2O_USE_OPENMP=ON -DOS3_USE_OPENMP=ON
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=syclcc -DCMAKE_CXX_FLAGS="-std=c++17 -O3 --hipsycl-targets=cuda:sm_86"
make -j4
