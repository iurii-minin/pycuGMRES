#include <cufft.h>
#include <cublas_v2.h>
#include <fstream>
#include <iomanip>
#include <curand_kernel.h>

#define WAVE_NUMBER 2*3.14f/(N/6.f)
#define Q 32
#define THREADS_PER_BLOCK N / Q
#define THREADS_PER_BLOCK_M THREADS_PER_BLOCK * 2
#define E0 1
#define ALPHA 3.14*0/180
#define EPSILON 2.25f
#define CHI (EPSILON-1)*WAVE_NUMBER*WAVE_NUMBER
#define PRECISION_TO_SAVE_DATA_TO_FILE 9

__global__ void extend_by_zeros_kernel(	cuComplex *dev_usual,  //For Gradient matvec (mask is absent)
					cuComplex *dev_extended,
					const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;
	cuComplex current;

	if ((i <  size_limit) && (j < size_limit ))
	{	
		unsigned int Ni = N * i;
		unsigned int index = Ni + j;
		unsigned int index_extended = index + Ni - i;
		if ((i < N) && (j < N))
		{
			current.x = CHI * dev_usual[index].x;
			current.y = CHI * dev_usual[index].y;
		}
		else
		{
			current.x = current.y = 0.f;
		}
		dev_extended[index_extended] = current;
	}
}
