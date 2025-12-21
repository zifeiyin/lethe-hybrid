#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi
cd build

source /home/saayzf/dealii-candi/configuration/enable.sh

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/saayzf/softwares/lethe-hybrid/platform
#cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/home/saayzf/softwares/lethe-hybrid/platform

make -j 12 
make install
