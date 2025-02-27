#!/bin/bash

# Check the first argument and set LD_LIBRARY_PATH and Makefile accordingly
if [ "$1" = "l" ]; then
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    MAKE_FILE="Makefile.local"
    echo "Using Conda's OpenCV library and Makefile.local."
elif [ "$1" = "p" ]; then
    export LD_LIBRARY_PATH=$(spack location -i opencv)/lib64:$LD_LIBRARY_PATH
    MAKE_FILE="Makefile.palmetto"
    echo "Using Spack's OpenCV library and Makefile.palmetto."
else
    echo "Invalid argument. Please use 'l' for Conda or 'p' for Spack."
    exit 1
fi

# Clean and build the project using the selected Makefile
make -f $MAKE_FILE clean
make -f $MAKE_FILE

# Run the blur program with the test image
# ./blur test_image.bmp