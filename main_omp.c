#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define IMAGE_HEIGHT 256
#define IMAGE_WIDTH 256
#define KERNEL_SIZE 3
#define LOOPS 10000

void conv2d(float *input, float *output, float *kernel, int height, int width, int kernel_size) {
  int kernel_size_half = kernel_size / 2;
  // параллелизация цикла как по y так и по x
  #pragma omp parallel for collapse(2) 
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float sum = 0.0;
      for (int fy = -kernel_size_half; fy <= kernel_size_half; fy++) {
        int in_y = y + fy;
        // имитация паддинга нулями
        if (in_y < 0 || in_y >= height) {
          continue;
        }

        for (int fx = -kernel_size_half; fx <= kernel_size_half; fx++) {
          int in_x = x + fx;
          if (in_x < 0 || in_x >= width) {
            continue;
          }

          int filter_index = (kernel_size_half + fy) * kernel_size + (kernel_size_half + fx);
          int image_index = in_y * width + in_x;
          sum += input[image_index] * kernel[filter_index];
        }
      }
      output[y * width + x] = sum;
    }
  }
}

void initialize_kernel(float *kernel, int kernel_size, const char* type) {
    
  for (int y = 0; y < kernel_size; y++) {
    for (int x = 0; x < kernel_size; x++) {
        if (type == "blur")
            kernel[y * kernel_size + x] = 1.0 / (kernel_size * kernel_size);
        else if (type == "edge")
        {
            kernel[y * kernel_size + x] = y == kernel_size / 2 && x == kernel_size / 2 ? 8.0 : -1.0;
        }
        
    }
  }
    
}

void print_image(float *image, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%f ", image[y * width + x]);
        }
        printf("\n");
    }
}

int main() {
  // для удобства матрицы хранятся в одномерном массиве и просто индексируются:
  // matr[y][x] <=> arr[y * col_size + x]
  float *image = (float*)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));
  float *kernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
  float *output = (float*)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));
  omp_set_dynamic(0);     
  omp_set_num_threads(3);
  printf("Num of threads: %d\n", omp_get_max_threads());
  // считывания картинки из файла
  const char *filename_in = "cat_tensor.txt";
  FILE *fin = fopen(filename_in, "r");
  if (fin == NULL) {
    fprintf(stderr, "Error opening file: %s\n", filename_in);
    return -1;
  }
  for (int y = 0; y < IMAGE_HEIGHT; y++) {
    for (int x = 0; x < IMAGE_WIDTH; x++) {
      fscanf(fin, "%f", &image[y * IMAGE_WIDTH + x]);
    }
  }
  fclose(fin);
  initialize_kernel(kernel, KERNEL_SIZE, "blur");

  double begin = omp_get_wtime();
  for (int i = 0; i < LOOPS; i++)
    conv2d(image, output, kernel, IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE);
  clock_t end = omp_get_wtime();
  double time_spent = (double)(end - begin);
  
  printf("Avg time spent: %f s", time_spent / LOOPS);
  
    // вывод картинки в файл
  const char *filename_out = "blur_cat.txt";
  FILE *fout = fopen(filename_out, "w"); 
  if (fout == NULL) {
    fprintf(stderr, "Error opening file: %s\n", filename_out);
    return -1;
  }
  for (int y = 0; y < IMAGE_HEIGHT; y++) {
    for (int x = 0; x < IMAGE_WIDTH; x++) {
      fprintf(fout, "%f ", output[y * IMAGE_WIDTH + x]);
    }
  }
  fclose(fout);


  free(image);
  free(kernel);
  free(output);
  return 0;
}
