if [ "$1" = "l" ]; then
    MAKE_FILE="Makefile.local"
    echo "Using Conda's OpenCV library and Makefile.local."
elif [ "$1" = "p" ]; then
    MAKE_FILE="Makefile.palmetto"
    echo "Using Spack's OpenCV library and Makefile.palmetto."
else
    echo "Invalid argument. Please use 'l' for Conda or 'p' for Spack."
    exit 1
fi
make -f $MAKE_FILE clean
rm -fr v100/