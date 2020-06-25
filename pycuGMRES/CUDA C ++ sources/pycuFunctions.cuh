struct devSubsidiary {
    cuComplex *dev_orthogonal_basis;
    int  *dev_info;
};


void pycuSetDevice(const unsigned int visible_device)
{
    cudacall(cudaSetDevice(visible_device));
}

void pycuInitSolution(
		        cuComplex *dev_solution,
		        const unsigned int N     
                     )
{
		dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
		dim3 threads(Q, Q);

		init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution, N);
		cudacheckSYN();
}

cufftHandle pycuGetPlan(const unsigned int N)
{
    cufftHandle plan;
    cufftcall(cufftPlan2d(&plan, 2 * N - 1, 2 * N - 1, CUFFT_C2C));
    return plan;
}


cublasHandle_t *pycuHandleBlas()
{
    cublasHandle_t *handle_p = (cublasHandle_t *) malloc(sizeof(cublasHandle_t));
    
    cublascall(cublasCreate_v2(handle_p));
    return handle_p;
}


cusolverDnHandle_t *pycuHandleSolverDn()
{
    cudaStream_t stream = NULL;
    cusolverDnHandle_t *cusolverH_p = (cusolverDnHandle_t *) malloc(sizeof(cusolverDnHandle_t));
    cusolvercall(cusolverDnCreate(cusolverH_p));
    cudacall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cusolvercall(cusolverDnSetStream(*cusolverH_p, stream));
    return cusolverH_p;
}


cuComplex *pycuGetGamma(cuComplex *dev_gamma_array, const unsigned int N, const cufftHandle plan) // typedef unsigned int cufftHandle;
{
    cufftcall(cufftExecC2C(plan, (cuComplex *)dev_gamma_array, (cuComplex *)dev_gamma_array, CUFFT_FORWARD));
    cudacheckSYN();

    return dev_gamma_array;
}

void pycuDestroyPlan(cufftHandle plan)
{
    cufftcall(cufftDestroy(plan));
}

void pycuFree(void *dev_var)
{
    cudacall(cudaFree(dev_var));
}

void pyFree(void *h_var)
{
    free(h_var);
}

void pycuDestroyBlas(cublasHandle_t *handle_p)
{
    cublascall(cublasDestroy_v2(*handle_p));
    free(handle_p);
}

void pycuDestroySolverDn(cusolverDnHandle_t *cusolverH_p)
{
    cusolvercall(cusolverDnDestroy(*cusolverH_p));
    //free(cusolverH_p);
}

void pycuSetPointerMode(cublasHandle_t *handle_p, cublasPointerMode_t mode)
{
    cublascall(cublasSetPointerMode(*handle_p, mode));
}

float pycuRelErr(	cuComplex *dev_solution, 
			cuComplex *dev_analytical_solution, 
			unsigned int N, 
			cublasHandle_t *handle_p)
{
    dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 threads(Q, Q);
    float h_result = 0.f;
    float h_norm_analytical_solution = 0.f;
    cuComplex *dev_C;

    cudacall(cudaMalloc(&dev_C, N * N * sizeof(cuComplex)));

    cublascall(cublasScnrm2(*handle_p, N * N,
                (const cuComplex *)dev_analytical_solution, 1, 
                (float  *)&h_norm_analytical_solution));

    cublascall( cublasScnrm2(*handle_p, N * N,
               (const cuComplex *)dev_solution, 1, (float  *)&h_result));
    
    A_minus_B_kernel <<< blocks, threads >>> (	(cuComplex *)dev_analytical_solution,
						(cuComplex *)dev_solution,
						(cuComplex *)dev_C,
						N);
    cudacheckSYN();


    cublascall(cublasScnrm2(*handle_p, N * N,
                        (const cuComplex *)dev_C, 1, (float  *)&h_result));

    h_result = h_result / h_norm_analytical_solution;

    cudacall(cudaFree((cuComplex *)dev_C));

    return h_result;
}


//extern "C" {
void *pycumalloc(unsigned int amount, size_t unit_size)
{
         void *dev_array;
         unsigned int size = amount * unit_size;
         cudacall(cudaMalloc(&dev_array, size));
         return dev_array;
}
//}

//extern "C" {
void pycuhost2gpu(void *h_array, void *dev_array, unsigned int amount, size_t unit_size)
{
        unsigned int size = amount * unit_size;
        cudacall(cudaMemcpy(dev_array, h_array, size, cudaMemcpyHostToDevice));
}
//}


//extern "C"{
void pycugpu2host(void *h_array, void *dev_array, unsigned int amount, size_t unit_size)
{
        unsigned int size = amount * unit_size;
        cudacall(cudaMemcpy(h_array, dev_array, size, cudaMemcpyDeviceToHost));
}
//}


const char *pycuGetSubsidiary(devSubsidiary *dev_subs, unsigned int N, unsigned int maxiter)
{
    cudaError_t err = cudaMalloc((void**)&(dev_subs->dev_orthogonal_basis), ((maxiter + 6) * N * N - 8 * N + 1) * sizeof(cuComplex));
    if(cudaSuccess != err)
    {
        return "no memory";
    }
    err = cudaMalloc((void**)&(dev_subs->dev_info), (maxiter + 1) * sizeof(int));
    if(cudaSuccess != err)
    {
        return "no memory";
    }
    return "success";
}

void pycuDestroySubsidiary(devSubsidiary *dev_subs)
{
    cudacall(cudaFree(dev_subs->dev_orthogonal_basis));
    cudacall(cudaFree(dev_subs->dev_info));
}

void pycuDeviceReset()
{
    cudaDeviceReset();
}

void pycuFFTC2C(cuComplex *dev_input, cuComplex *dev_output, cufftHandle plan)
{
    cufftcall(cufftExecC2C(plan, (cuComplex *)dev_input, (cuComplex *)dev_output, CUFFT_FORWARD));
    cudacheckSYN();
}

void pycuGxFFTmatvec_grad(	
			cuComplex *dev_gamma_array, // For gradient matvec (dev_mask is absent)
			cuComplex *dev_solution,
			cuComplex *dev_matmul_out_extended,
			cufftHandle plan,
			const unsigned int N)
{
	G_x_fft_matvec(	(cuComplex  *)dev_gamma_array, // For gradient matvec (dev_mask is absent)
			(cuComplex  *)dev_solution,
			(cuComplex  *)dev_matmul_out_extended,
			(cufftHandle )plan,
										N);
}

void pycu2Dto1Dgrad( //For gradient computations
					cuComplex *dev_solution, 
					cuComplex *dev_new_z_extended, 
					float *dev_gradient, 
					const unsigned int h_index_of_max,
					const unsigned int N)
{
	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);

	_2D_to_1D_kernel <<< blocks, threads >>> (
					(cuComplex    *)dev_solution, 
					(cuComplex    *)dev_new_z_extended, 
					(float        *)dev_gradient, 
													h_index_of_max,
													N);
	cudacheckSYN();
}
