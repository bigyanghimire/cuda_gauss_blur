# loop through the Test_Images folder
# images are img_256, img_1024, img_2048.bmp
# go through each images and do this
#     make this ncu -o "$GPU_NAME+t_$NUMBER_OF_THREADS+img_name" ./blur Test_Images/img_name $GPU_NAME/$NUMBER_OF_THREADS/img_name_result.bmp
# # ncu -o profile ./blur test_32.bmp result_32.bmp


#!/bin/bash

# Set your GPU name and number of threads
GPU_NAME=$1
NUMBER_OF_THREADS=$2  # Adjust as needed

# Directory containing images
IMAGE_DIR="Test_Images"

# Loop through each image
for image in "$IMAGE_DIR"/*; do
    # Extract the filename without the path
    img_name=$(basename "$image")
    img_first_name=$(basename "$image" | cut -d. -f1)


    # Construct output filename
    output_name="${GPU_NAME}+t_${NUMBER_OF_THREADS}+${img_name}"
    
    # Construct result path
    result_path="${GPU_NAME}/${NUMBER_OF_THREADS}/${img_first_name}_result.bmp"

    # Create directory if not exists
    mkdir -p "$(dirname "$result_path")"

    # Run the profiling command
    ncu -o "$output_name" ./blur "$image" "$result_path"
done
