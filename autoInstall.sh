#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/saayzf/softwares/lethe-hybrid/platform

make -j 4 
make install
