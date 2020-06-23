void pycuGMRESrmk(	
                    bool *dev_mask,
                    cuComplex *dev_solution,
										const bool for_gradient,
										const unsigned int h_index_of_max,
										unsigned int maxiter,
										unsigned int *GMRES_n,
										float *dev_actual_residual,
										bool *h_res_vs_tol_p,
										const unsigned int N,
                    cuComplex *dev_gamma_array,
                    const cufftHandle plan,
                    cublasHandle_t *handle_p,
                    cusolverDnHandle_t *cusolverH_p,
                    devSubsidiary *dev_subs,
                    timespec *h_computation_times
               )
{		
	char buffer[1024];
	float tolerance = 0.001f;//0.2f;

	cuComplex **p_h_gamma_arrays = (cuComplex **) malloc((1) * sizeof(cuComplex *));



	p_h_gamma_arrays[0] = (cuComplex *)malloc((2 * N - 1) * (2 * N - 1) * sizeof(cuComplex));

	sprintf(buffer, "/media/linux/4db3d51d-3503-451d-aff7-07e3ce95927e/Archive/Input/G_prep_%u.txt", N);
	get_array_C_to_CPU((cuComplex *)p_h_gamma_arrays[0], (const char *)buffer);



	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 threads(Q, Q);

	cuComplex *h_gamma_array = p_h_gamma_arrays[0];

	cudacall(cudaMemcpy(dev_gamma_array, h_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyHostToDevice));

	cufftcall(cufftExecC2C(plan, (cuComplex *)dev_gamma_array, (cuComplex *)dev_gamma_array, CUFFT_FORWARD));
	cudacheckSYN();

		pycuInitSolution((cuComplex *)dev_solution, N);


		pycuGMRES(	
					  (bool *)dev_mask,
					  (cuComplex *)dev_solution,
					  for_gradient,
					  h_index_of_max,
					  maxiter,
					  tolerance,
					  (unsigned int *)GMRES_n,
					  (float *)dev_actual_residual,
					  (bool *)h_res_vs_tol_p,
					  N,
					  (cuComplex *)dev_gamma_array,
					  plan,
					  (cublasHandle_t *)handle_p,
					  (cusolverDnHandle_t *)cusolverH_p,
					  (devSubsidiary *)dev_subs,
					  (timespec *)h_computation_times
			);

}
