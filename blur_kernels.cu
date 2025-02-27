#include "./gaussian_kernel.h"

/*
The actual gaussian blur kernel to be implemented by
you. Keep in mind that the kernel operates on a
single channel.
 */
__global__ void gaussianBlur(unsigned char *d_in, unsigned char *d_out,
                             const int rows, const int cols, float *d_filter, const int filterWidth)
{
        int halfWidth = filterWidth / 2;
        float pixelSum = 0.0f;
        float weightSum = 0.0f; // For normalization
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int idx = cols * j + i;
        // if (idx < 1000 && idx > 989)
        // {
        //         printf("the red value is for thread %d with value %d\n", idx, d_in[idx]);
        // }

        for (int fr = -halfWidth; fr <= halfWidth; ++fr)
        {
                for (int fc = -halfWidth; fc <= halfWidth; ++fc)
                {
                        int imageRow = j + fr;
                        int imageCol = i + fc;

                        // Check bounds (to avoid accessing outside image)
                        if (imageRow >= 0 && imageRow < rows && imageCol >= 0 && imageCol < cols)
                        {
                                int imageIndex = imageRow * cols + imageCol;
                                int filterIndex = (fr + halfWidth) * filterWidth + (fc + halfWidth);
                                // if (idx == 5000)
                                // {
                                //         printf("the output red gpu image index %d with value fi;lter index %d\n", imageIndex, filterIndex);
                                // }
                                pixelSum += d_filter[filterIndex] * d_in[imageIndex];
                                weightSum += d_filter[filterIndex];
                        }
                }
        }
        int output_index = cols * j + i; // 2D (i,j) to 1D index
        d_out[output_index] = static_cast<unsigned char>(pixelSum / weightSum);
        if (output_index == 2)
        {
                printf("the output red gpu  %d with value %d\n", output_index, d_out[output_index]);
        }
}

/*
  Given an input RGBA image separate
  that into appropriate rgba channels.
 */
__global__ void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b,
                                 const int rows, const int cols)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < cols && j < rows)
        {

                int idx = cols * j + i;
                uchar4 pixel = d_imrgba[idx];

                unsigned char red = pixel.x;
                unsigned char green = pixel.y;
                unsigned char blue = pixel.z;
                d_r[idx] = red;
                d_g[idx] = green;
                d_b[idx] = blue;
        }
}

/*
  Given input channels combine them
  into a single uchar4 channel.

  You can use some handy constructors provided by the
  cuda library i.e.
  make_int2(x, y) -> creates a vector of type int2 having x,y components
  make_uchar4(x,y,z,255) -> creates a vector of uchar4 type x,y,z components
  the last argument being the transperency value.
 */
__global__ void recombineChannels(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, uchar4 *d_orgba,
                                  const int rows, const int cols)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < cols && j < rows)
        {
                int idx = cols * j + i;
                uchar4 pixel;
                pixel.x = d_b[idx]; // Assign red channel
                pixel.y = d_g[idx]; // Assign green channel
                pixel.z = d_r[idx]; // Assign blue channel
                pixel.w = 255;      // Set alpha channel to 255 (fully opaque)
                d_orgba[idx] = pixel;
                // if (idx > 0 && idx < 10)
                // {
                //         printf("the output red blurred for thread %d with value %d\n", idx, d_r[idx]);
                // }
                if (idx == 2)
                {
                        printf("the final gpu output for   %d with value %d\n", idx, d_orgba[idx]);
                }
        }
}

void your_gauss_blur(uchar4 *d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols,
                     unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue,
                     unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
                     float *d_filter, int filterWidth, int num_threads)
{

        int threads_per_block = num_threads;
        dim3 gridSize((cols + threads_per_block - 1) / threads_per_block, (rows + threads_per_block - 1) / threads_per_block, 1);
        dim3 blockSize(threads_per_block, threads_per_block, 1);

        // dim3 blockSize(1,1,1);
        // dim3 gridSize(1,1,1);

        separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
}
