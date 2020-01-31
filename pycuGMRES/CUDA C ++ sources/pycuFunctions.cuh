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


cuComplex *pycuGetGamma(const cuComplex *h_gamma_array, const unsigned int N, const cufftHandle plan) // typedef unsigned int cufftHandle;
{
    cuComplex *dev_gamma_array;
            //	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);
            //	cudacall(cudaMemcpy(h_gamma_array, dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyDeviceToHost));
            //==================================== Begin: get_gamma_array connected to MKL 2D Green's function values in Bessel function =========================
    cudacall(cudaMalloc((void**)&dev_gamma_array,  (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex)));
    cudacall(cudaMemcpy(dev_gamma_array, h_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyHostToDevice));

    cufftcall(cufftExecC2C(plan, (cuComplex *)dev_gamma_array, (cuComplex *)dev_gamma_array, CUFFT_FORWARD));
    cudacheckSYN();
            //==================================== End: get_gamma_array connected to MKL 2D Green's function values in Bessel function =========================
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

float pycuRelErr(cuComplex *dev_solution, cuComplex *dev_analytical_solution, unsigned int N, cublasHandle_t *handle_p)
{
    float h_result = 0.f;
    float h_norm_analytical_solution = 0.f;
    cuComplex alpha;
    alpha.x = -1.f;
    alpha.y = 0.f;

    cublascall(cublasScnrm2(*handle_p, N * N,
                (const cuComplex *)dev_analytical_solution, 1, 
                (float  *)&h_norm_analytical_solution));

    cublascall( cublasScnrm2(*handle_p, N * N,
               (const cuComplex *)dev_solution, 1, (float  *)&h_result));

    fprintf(stderr, "Norm of solution:\t%f\n", h_result);


    cublascall( cublasCaxpy(*handle_p, N * N,
               (const cuComplex *)&alpha,
               (const cuComplex *)dev_analytical_solution, 1,
               (cuComplex *)dev_solution, 1));


    cublascall(cublasScnrm2(*handle_p, N * N,
                        (const cuComplex *)dev_solution, 1, (float  *)&h_result));

    fprintf(stderr, "Norm of diff:\t%f\n", h_result);

    h_result = h_result / h_norm_analytical_solution;

    

    fprintf(stderr, "relative_error:\t%f\n", h_result);

    return h_result;
}


//extern "C" {
void *pycumalloc(unsigned int amount, size_t unit_size)
{
         bool *dev_array;
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
