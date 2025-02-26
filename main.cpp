#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "utils.h"
#include "gaussian_kernel.h"

/*
 * Compute if the two images are correctly
 * computed. The reference image can
 * either be produced by a software or by
 * your own serial implementation.
 * */
void checkApproxResults(unsigned char *ref, unsigned char *gpu, size_t numElems)
{

    for (int i = 0; i < numElems; i++)
    {
        std::cout << "GPU:: " << gpu[i] << "\n";
        std::cout << "CPU:: " << ref[i] << "\n";
        if (ref[i] - gpu[i] > 1e-5)
        {
            std::cerr << "Error at position " << i << "\n";

            std::cerr << "Reference:: " << std::setprecision(17) << +ref[i] << "\n";
            std::cerr << "GPU:: " << +gpu[i] << "\n";

            exit(1);
        }
    }
}

void checkResult(const std::string &reference_file, const std::string &output_file, float eps)
{
    cv::Mat ref_img, out_img;

    ref_img = cv::imread(reference_file, -1);
    out_img = cv::imread(output_file, -1);

    unsigned char *refPtr = ref_img.ptr<unsigned char>(0);
    unsigned char *oPtr = out_img.ptr<unsigned char>(0);

    checkApproxResults(refPtr, oPtr, ref_img.rows * ref_img.cols * ref_img.channels());
    std::cout << "PASSED!\n";
}

void gaussian_blur_filter(float *arr, const int f_sz, const float f_sigma = 15)
{
    float filterSum = 0.f;
    float norm_const = 0.0; // normalization const for the kernel

    for (int r = -f_sz / 2; r <= f_sz / 2; r++)
    {
        for (int c = -f_sz / 2; c <= f_sz / 2; c++)
        {
            float fSum = expf(-(float)(r * r + c * c) / (2 * f_sigma * f_sigma));
            arr[(r + f_sz / 2) * f_sz + (c + f_sz / 2)] = fSum;
            filterSum += fSum;
        }
    }

    norm_const = 1.f / filterSum;

    for (int r = -f_sz / 2; r <= f_sz / 2; ++r)
    {
        for (int c = -f_sz / 2; c <= f_sz / 2; ++c)
        {
            arr[(r + f_sz / 2) * f_sz + (c + f_sz / 2)] *= norm_const;
        }
    }
}

// Serial implementations of kernel functions
void serialGaussianBlur(unsigned char *in, unsigned char *out, const int rows, const int cols,
                        float *filter, const int filterWidth)
{
    int halfWidth = filterWidth / 2; // Half width of filter

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            float pixelSum = 0.0f;
            float weightSum = 0.0f;

            for (int fr = -halfWidth; fr <= halfWidth; ++fr)
            {
                for (int fc = -halfWidth; fc <= halfWidth; ++fc)
                {
                    int imageRow = i + fr;
                    int imageCol = j + fc;

                    // Check bounds (to avoid accessing outside image)
                    if (imageRow >= 0 && imageRow < rows && imageCol >= 0 && imageCol < cols)
                    {
                        int imageIndex = imageRow * cols + imageCol;
                        int filterIndex = (fr + halfWidth) * filterWidth + (fc + halfWidth);
                        // if (cols * i + j == 5000)
                        // {
                        //     printf("the output red cpu image index %d with value fi;lter index %d\n", imageIndex, filterIndex);
                        // }
                        pixelSum += filter[filterIndex] * in[imageIndex];
                        weightSum += filter[filterIndex];
                    }
                }
            }

            int output_index = cols * i + j;

            out[output_index] = static_cast<unsigned char>(pixelSum / weightSum);
            if (output_index == 2)
            {
                printf("the output red cpu  %d with value %d\n", output_index, out[output_index]);
            }
        }
    }
}
void serialSeparateChannels(uchar4 *imrgba, unsigned char *r, unsigned char *g, unsigned char *b,
                            const int rows, const int cols)
{
    printf("total pxiels are are %d\n", rows * cols);
    for (int i = 0; i < rows * cols; ++i)
    {

        uchar4 pixel = imrgba[i];

        unsigned char red = pixel.x;
        unsigned char green = pixel.y;
        unsigned char blue = pixel.z;
        r[i] = red;
        g[i] = green;
        b[i] = blue;
    }
}

void serialRecombineChannels(unsigned char *r, unsigned char *g, unsigned char *b, uchar4 *orgba,
                             const int rows, const int cols)
{
    int totalPixels = rows * cols;

    for (int i = 0; i < totalPixels; ++i)
    {
        uchar4 pixel;
        pixel.x = b[i]; // Assign red channel
        pixel.y = g[i]; // Assign green channel
        pixel.z = r[i]; // Assign blue channel
        pixel.w = 255;  // Set alpha channel to 255 (fully opaque)
        orgba[i] = pixel;
        if (i == 2)
        {
            printf("the final cpu output for   %d with value %d\n", i, orgba[i]);
        }
    }
}

int main(int argc, char const *argv[])
{

    uchar4 *h_in_img, *h_o_img; // pointers to the actual image input and output pointers
    uchar4 *d_in_img, *d_o_img, *r_o_img;

    //    unsigned char *h_red, *h_blue, *h_green;
    unsigned char *d_red, *d_blue, *d_green;

    float *h_filter, *d_filter;
    cv::Mat imrgba, o_img, t_img;

    const int fWidth = 9;
    const float fDev = 16;
    std::string infile;
    std::string outfile;
    std::string reference;

    switch (argc)
    {
    case 2:
        infile = std::string(argv[1]);
        outfile = "blurred_gpu.bmp";
        reference = "blurred_serial.bmp";
        break;
    case 3:
        infile = std::string(argv[1]);
        outfile = std::string(argv[2]);
        reference = "blurred_serial.bmp";
        break;
    case 4:
        infile = std::string(argv[1]);
        outfile = std::string(argv[2]);
        reference = std::string(argv[3]);
        break;
    default:
        std::cerr << "Usage ./gblur <in_image> <out_image> <reference_file> \n";
        exit(1);
    }

    // preprocess
    cv::Mat img = cv::imread(infile.c_str(), cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Image file couldn't be read, exiting\n";
        exit(1);
    }

    cv::cvtColor(img, imrgba, cv::COLOR_BGR2RGBA);

    o_img.create(img.rows, img.cols, CV_8UC4);
    t_img.create(img.rows, img.cols, CV_8UC4);

    const size_t numPixels = img.rows * img.cols;

    h_in_img = (uchar4 *)imrgba.ptr<unsigned char>(0); // pointer to input image
    h_o_img = (uchar4 *)o_img.ptr<unsigned char>(0);   // pointer to output image
    r_o_img = (uchar4 *)t_img.ptr<unsigned char>(0);   // pointer to reference output image

    // allocate the memories for the device pointers
    // unsigned char *h_red, *h_blue, *h_green;

    unsigned char *h_red = new unsigned char[img.rows * img.cols];
    unsigned char *h_blue = new unsigned char[img.rows * img.cols];
    unsigned char *h_green = new unsigned char[img.rows * img.cols];

    unsigned char *d_red_blurred = new unsigned char[img.rows * img.cols];
    unsigned char *d_green_blurred = new unsigned char[img.rows * img.cols];
    unsigned char *d_blue_blurred = new unsigned char[img.rows * img.cols];
    // filter allocation
    h_filter = new float[fWidth * fWidth];
    gaussian_blur_filter(h_filter, fWidth, fDev); // create a filter of 9x9 with std_dev = 0.2
                                                  // Serial implementation
    serialSeparateChannels(h_in_img, h_red, h_green, h_blue, img.rows, img.cols);

    // for (int i = 60000; i < 60200; ++i)
    // {
    //     printf("serial Pixel %d: r=%d\n", i, h_blue[i]);
    // }
    serialGaussianBlur(h_red, d_red_blurred, img.rows, img.cols, h_filter, fWidth);
    serialGaussianBlur(h_green, d_green_blurred, img.rows, img.cols, h_filter, fWidth);
    serialGaussianBlur(h_blue, d_blue_blurred, img.rows, img.cols, h_filter, fWidth);
    // for (int i = 0; i < 10; ++i)
    // {
    //     printf("Pixel %d: r=%d, g=%d, b=%d\n", i, d_red_blurred[i], d_green_blurred[i], d_blue_blurred[i]);
    // }
    serialRecombineChannels(d_red_blurred, d_green_blurred, d_blue_blurred, r_o_img, img.rows, img.cols);
    printf("the output red cpu %d with value %d\n", 1, d_red_blurred[15]);
    // for (int i = 0; i < 10; ++i)
    // {
    //     printf("the output bluured %d with value %d\n", i, d_red_blurred[i]);
    // }
    // for (int i = 1000; i < 1010; ++i)
    // {
    //     printf("the output values %d with value %d\n", i, r_o_img[i]);
    // }
    // printArray<float>(h_filter, fWidth * fWidth); // printUtility.

    // GPU implementation
    // copy the image and filter over to GPU here
    checkCudaErrors(cudaMalloc((void **)&d_in_img, sizeof(uchar4) * img.rows * img.cols));
    checkCudaErrors(cudaMalloc((void **)&d_o_img, sizeof(uchar4) * img.rows * img.cols));
    checkCudaErrors(cudaMalloc((void **)&d_red, sizeof(unsigned char) * img.rows * img.cols));
    checkCudaErrors(cudaMalloc((void **)&d_green, sizeof(unsigned char) * img.rows * img.cols));
    checkCudaErrors(cudaMalloc((void **)&d_blue, sizeof(unsigned char) * img.rows * img.cols));
    checkCudaErrors(cudaMalloc((void **)&d_red_blurred, sizeof(unsigned char) * img.rows * img.cols));
    checkCudaErrors(cudaMalloc((void **)&d_green_blurred, sizeof(unsigned char) * img.rows * img.cols));
    checkCudaErrors(cudaMalloc((void **)&d_blue_blurred, sizeof(unsigned char) * img.rows * img.cols));
    checkCudaErrors(cudaMalloc((void **)&d_filter, sizeof(float) * img.rows * img.cols));

    // Copy to device memory
    checkCudaErrors(cudaMemcpy(d_in_img, h_in_img, sizeof(uchar4) * img.rows * img.cols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * fWidth * fWidth, cudaMemcpyHostToDevice));
    // kernel launch code
    your_gauss_blur(d_in_img, d_o_img, img.rows, img.cols, d_red, d_green, d_blue,
                    d_red_blurred, d_green_blurred, d_blue_blurred, d_filter, fWidth);

    // memcpy the output image to the host side.
    checkCudaErrors(cudaMemcpy(h_o_img, d_o_img, sizeof(uchar4) * img.rows * img.cols, cudaMemcpyDeviceToHost));
    // perform serial memory allocation and function calls, final output should be stored in *r_o_img
    //  ** there are many ways to perform timing in c++ such as std::chrono **

    // create the image with the output data

    cv::Mat output(img.rows, img.cols, CV_8UC4, (void *)h_o_img); // generate GPU output image.
    bool suc = cv::imwrite(outfile.c_str(), output);
    if (!suc)
    {
        std::cerr << "Couldn't write GPU image!\n";
        exit(1);
    }
    cv::Mat output_s(img.rows, img.cols, CV_8UC4, (void *)r_o_img); // generate serial output image.
    suc = cv::imwrite(reference.c_str(), output_s);
    if (!suc)
    {
        std::cerr << "Couldn't write serial image!\n";
        exit(1);
    }

    // check if the caclulation was correct to a degree of tolerance

    checkResult(reference, outfile, 1e-5);

    // free any necessary memory.
    cudaFree(d_in_img);
    cudaFree(d_o_img);
    cudaFree(d_red);
    cudaFree(d_green);
    // cudaFree(d_grey);
    delete[] h_filter;
    return 0;
}
