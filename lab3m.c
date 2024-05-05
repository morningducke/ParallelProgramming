#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define IMAGE_HEIGHT 256
#define IMAGE_WIDTH 256
#define KERNEL_SIZE 3
#define LOOPS 10000
void conv2d(float *input, float *output, float *kernel, int start, int end, int height, int width, int kernel_size)
{
    int kernel_size_half = kernel_size / 2;
    int y = start / width;
    int end_y = end / width;
    for (; y < height && y < end_y; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float sum = 0.0;
            for (int fy = -kernel_size_half; fy <= kernel_size_half; fy++)
            {
                int in_y = y + fy;
                // // имитация паддинга
                // if (in_y < 0 || in_y >= end_y)
                // {
                //     sum += 1 * kernel[0];
                //     continue;
                // }
                for (int fx = -kernel_size_half; fx <= kernel_size_half; fx++)
                {
                    int in_x = x + fx;
                    int filter_index = (kernel_size_half + fy) * kernel_size +
                                       (kernel_size_half + fx);

                    // паддинг
                    if (in_x < 0 || in_x >= width || in_y < 0 || in_y >= end_y)
                    {
                        sum += 1. * kernel[filter_index];
                        continue;
                    }
                    
                    int image_index = in_y * width + in_x;
                    sum += input[image_index] * kernel[filter_index];
                }
            }
            output[y * width + x] = sum;
        }
    }
}
void initialize_kernel(float *kernel, int kernel_size, const char *type)
{
    for (int y = 0; y < kernel_size; y++)
    {
        for (int x = 0; x < kernel_size; x++)
        {
            if (type == "blur")
                kernel[y * kernel_size + x] = 1.0 / (kernel_size * kernel_size);
            else if (type == "edge")
            {
                kernel[y * kernel_size + x] = y == kernel_size / 2 && x ==
                                                                          kernel_size / 2
                                                  ? 8.0
                                                  : -1.0;
            }
        }
    }
}

void initialize_image(float *image, int height, int width) {
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            image[y * width + x] = 1.;
            // fscanf(fin, "%f", &image[y * IMAGE_WIDTH + x]);
        }
    }
}

void print_image(float *image, int height, int width)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf("%f ", image[y * width + x]);
        }
        printf("\n");
    }
}
int main(int argc, char *argv[])
{

    int rank, proc_count;
    double start_time, end_time;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    if (IMAGE_HEIGHT * IMAGE_WIDTH % proc_count != 0) {
        fprintf(stderr, "Image can't be split into %d chunks", proc_count);
    }
    int chunk_size = IMAGE_HEIGHT * IMAGE_WIDTH / proc_count;
    // для удобства матрицы хранятся в одномерном массиве и просто индексируются:
    // matr[y][x] <=> arr[y * col_size + x]

    // printf("rank: %d \n", rank);
    float *image, *kernel, *output;
    if (rank == 0) {
        image = (float *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));
        kernel = (float *)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
        output = (float *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));
    
        // printf("chunk size: %d \n", chunk_size);
        // printf("proc count: %d \n", proc_count);
    
        initialize_image(image, IMAGE_HEIGHT, IMAGE_WIDTH);
        initialize_kernel(kernel, KERNEL_SIZE, "blur");

        start_time = MPI_Wtime();

        for (int i = 0; i < LOOPS; i++) 
        {
            // распределение данных между процессорами
            for (int p = 1; p < proc_count; p++) {
                MPI_Send(&image[p * chunk_size], chunk_size, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
            }
            conv2d(image, output, kernel, 0, chunk_size, IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE);
            // cбор результатов
            for (int p = 1; p < proc_count; p++) {
                MPI_Recv(&output[p * chunk_size], chunk_size, MPI_FLOAT, p, 0, MPI_COMM_WORLD, &status);
            }
        }
        end_time = MPI_Wtime();
        // вывод результата
        // print_image(output, IMAGE_HEIGHT, IMAGE_WIDTH);
        printf("Avg time spent: %f s\n", (end_time - start_time) / LOOPS);
    }
    else {
        image = (float *)malloc(chunk_size * sizeof(float));
        kernel = (float *)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
        output = (float *)malloc(chunk_size * sizeof(float));

        initialize_kernel(kernel, KERNEL_SIZE, "blur");

        for (int i = 0; i < LOOPS; i++)
        {
            MPI_Recv(image, chunk_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
            conv2d(image, output, kernel, 0, chunk_size, IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE);
            MPI_Send(output, chunk_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }


    free(image);
    free(kernel);
    free(output);
    MPI_Finalize();
    return 0;
}