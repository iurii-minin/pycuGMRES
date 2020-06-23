void pycuGMRESrmk(	
                    bool *dev_mask,
                    cuComplex *dev_solution,
										const bool for_gradient,
										const unsigned int h_index_of_max,
										unsigned int maxiter,
										const float tolerance,
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
	float tolerance_internal = 0.001f;//0.2f;

	unsigned int rep_st = 0;
	unsigned int rep_en = 0;

	unsigned int pow_st = 10;
	unsigned int pow_en = 10;

	unsigned int min_maxiter = 30;
	unsigned int max_maxiter = 30;

	unsigned int *n_timestamps_array = get_n_timestamps_array_improved((unsigned int)max_maxiter + 1);


	devSubsidiary dev_subs_internal[1];


	cuComplex **p_h_anal_sols = (cuComplex **) malloc((pow_en - pow_st + 1) * sizeof(cuComplex *));
	bool **p_h_masks = (bool **) malloc((pow_en - pow_st + 1) * sizeof(bool *));
	cuComplex **p_h_gamma_arrays = (cuComplex **) malloc((pow_en - pow_st + 1) * sizeof(cuComplex *));

	for (unsigned int pow_cur = pow_st; pow_cur < pow_en + 1; pow_cur ++)
	{
		unsigned int N = 1 << pow_cur;
		p_h_anal_sols[pow_cur - pow_st] = (cuComplex *) malloc( N * N * sizeof(cuComplex) );

		std::string line;
		sprintf(buffer, "/media/linux/4db3d51d-3503-451d-aff7-07e3ce95927e/Archive/Input/analytical_solution_%u.txt", N);
		std::ifstream analytical_solution_file (buffer);//Python_analytical_solution_%u
		if (analytical_solution_file.is_open())
		{
			unsigned int index = 0;
			while ( getline (analytical_solution_file, line) )
			{
				std::istringstream in_string_stream(line);

				in_string_stream >> p_h_anal_sols[pow_cur - pow_st][index].x >> p_h_anal_sols[pow_cur - pow_st][index].y;

				index++;
	
			}
			analytical_solution_file.close();
		}
		else
		{
			fprintf(stderr, "Unable to open file: %s\n", buffer);
			exit(1);
		}


		p_h_masks[pow_cur - pow_st] = (bool *) malloc(N * N * sizeof(bool));

		sprintf(buffer, "/media/linux/4db3d51d-3503-451d-aff7-07e3ce95927e/Archive/Input/cylinder_%u.txt", N);
		std::ifstream myfile (buffer);
		if (myfile.is_open())
		{
			unsigned int index = 0;
			while ( getline (myfile,line) )
			{
				p_h_masks[pow_cur - pow_st][index++] = (line == "1");
			}
			myfile.close();
		}
		else {
			fprintf(stderr, "Unable to open file: %s\n", buffer);
			exit(1);
		}


		p_h_gamma_arrays[pow_cur - pow_st] = (cuComplex *)malloc((2 * N - 1) * (2 * N - 1) * sizeof(cuComplex));
	
		sprintf(buffer, "/media/linux/4db3d51d-3503-451d-aff7-07e3ce95927e/Archive/Input/G_prep_%u.txt", N);
		get_array_C_to_CPU((cuComplex *)p_h_gamma_arrays[pow_cur - pow_st], (const char *)buffer);
	}


	for (unsigned int repetition_i = rep_st; repetition_i < rep_en + 1; repetition_i ++)
	{	// int maxiter = 28;
		for (unsigned int maxiter = min_maxiter; maxiter < max_maxiter + 1; maxiter ++)
		{
			for (unsigned int pow_cur = pow_st; pow_cur < pow_en + 1; pow_cur = pow_cur + 5) //Characteristic size of square matrix
			{
				unsigned int N = 1 << pow_cur;

				fprintf(stderr, "%i\n", N);
	
				dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
				dim3 threads(Q, Q);
				cufftHandle plan;
				cublasHandle_t handle;
				cublascall(cublasCreate_v2(&handle));
				cufftcall(cufftPlan2d(&plan, 2 * N - 1, 2 * N - 1, CUFFT_C2C));
				cudaStream_t stream = NULL;
				cusolverDnHandle_t cusolverH = NULL;
				cusolvercall(cusolverDnCreate(&cusolverH));
				cudacall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
				cusolvercall(cusolverDnSetStream(cusolverH, stream));

				bool *dev_mask;
				bool *h_mask = p_h_masks[pow_cur - pow_st];
				bool h_res_vs_tol = true;
				cuComplex *h_gamma_array = p_h_gamma_arrays[pow_cur - pow_st];
				cuComplex *h_analytical_solution = p_h_anal_sols[pow_cur - pow_st];
				cuComplex *dev_gamma_array;
				cuComplex *dev_analytical_solution;
				cuComplex *dev_solution;
				float *dev_actual_residual;
				float h_result = 0.f;
				float h_norm_analytical_solution = 0.f;
				unsigned int GMRES_n = 0;
				timespec *h_computation_times = (timespec *) malloc(n_timestamps_array[maxiter] * sizeof(timespec));
				cudacall(cudaSetDevice(0));

				cudacall(cudaMalloc((void**)&dev_mask, N * N * sizeof(bool)));
				cudacall(cudaMalloc((void**)&dev_solution, N * N * sizeof(cuComplex)));
				cudacall(cudaMalloc((void**)&dev_analytical_solution, N * N * sizeof(cuComplex)));




				cudacall(cudaMemcpy(dev_analytical_solution, h_analytical_solution, N * N * sizeof(cuComplex), cudaMemcpyHostToDevice));


				cublascall(cublasScnrm2(handle, N * N,
							(const cuComplex *)dev_analytical_solution, 1, 
							(float  *)&h_norm_analytical_solution));


				cudacall(cudaMemcpy(dev_mask, h_mask, N * N * sizeof(bool), cudaMemcpyHostToDevice));

			//	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);
			//	cudacall(cudaMemcpy(h_gamma_array, dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyDeviceToHost));
			//==================================== Begin: get_gamma_array connected to MKL 2D Green's function values in Bessel function =========================
				cudacall(cudaMalloc((void**)&dev_gamma_array,  (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex)));
				cudacall(cudaMemcpy(dev_gamma_array, h_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyHostToDevice));

				cufftcall(cufftExecC2C(plan, (cuComplex *)dev_gamma_array, (cuComplex *)dev_gamma_array, CUFFT_FORWARD));
				cudacheckSYN();
			//==================================== End: get_gamma_array connected to MKL 2D Green's function values in Bessel function =========================

				time_t clock_time;
				float diff_time = 0.f;
				float diff_average = 0.f;
				cuComplex alpha;
				alpha.x = -1.f;
				alpha.y = 0.f;
				const cuComplex *p_alpha = &alpha;

				{

					cudacall(cudaMalloc((void**)&dev_actual_residual, (maxiter + 1) * sizeof(float)));

					const char *allocation_result = pycuGetSubsidiary(
								(devSubsidiary *)dev_subs_internal, 
								N, 
								maxiter);

					fprintf(stderr, "Allocation memory: %s\n", allocation_result);

					cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
					fprintf(stderr, "maxiter = %i\trepetition_i = %i\n", maxiter, repetition_i);

					init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution, N);
					cudacheckSYN();

					memset(h_computation_times, 0, n_timestamps_array[maxiter] * sizeof(timespec));

					clock_time = clock();
 
					pycuGMRES(	
							    (bool *)dev_mask,
							    (cuComplex *)dev_solution,
							    false,
							    0,
							    maxiter,
							    tolerance_internal,
							    (unsigned int *)&GMRES_n,
							    (float *)dev_actual_residual,
							    (bool *)&h_res_vs_tol,
							    N,
							    (cuComplex *)dev_gamma_array,
							    plan,
							    (cublasHandle_t *)&handle,
							    (cusolverDnHandle_t *)&cusolverH,
							    (devSubsidiary *)dev_subs,
							    (timespec *)h_computation_times
						);

					diff_time = (float)(clock() - clock_time) / (float)(CLOCKS_PER_SEC);

					pycuDestroySubsidiary((devSubsidiary *)dev_subs_internal);
				}

				{
					fprintf(stderr, "Files writing\n");
		
					sprintf(buffer, "time_%u/solution_sample", N);
					save_test_GPU((char *)buffer, (cuComplex *)dev_solution, maxiter * 100 + repetition_i, N * N);
					fprintf(stderr, "diff_time = %f\n", diff_time);

					sprintf(buffer, "time_%u/maxiter", N);
					save_test_F_CPU((char *)buffer, (float *)&diff_time, maxiter * 100 + repetition_i, 1);
					sprintf(buffer, "time_%u/residual", N);
					save_test_F_GPU((char *)buffer, (float *)dev_actual_residual + GMRES_n, maxiter * 100 + repetition_i, 1);
					sprintf(buffer, "time_%u/times", N);
					save_test_timespec_CPU((char *)buffer, (timespec *)h_computation_times, maxiter * 100 + repetition_i, n_timestamps_array[maxiter]);

					cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

					cublascall(cublasScnrm2(handle, N * N,
								    (const cuComplex *)dev_solution, 1, (float  *)&h_result));

					fprintf(stderr, "Norm of solution = %f\n", h_result);


					cublascall(cublasCaxpy(handle, N * N,
								   (const cuComplex *)p_alpha,
								   (const cuComplex *)dev_analytical_solution, 1,
								   (cuComplex *)dev_solution, 1));


					cublascall(cublasScnrm2(handle, N * N,
								    (const cuComplex *)dev_solution, 1, (float  *)&h_result));

					fprintf(stderr, "Norm of diff = %f\n", h_result);

					h_result = h_result / h_norm_analytical_solution;

					fprintf(stderr, "File relative_error writing\t%f\n", h_result);
					sprintf(buffer, "time_%u/relative_error", N);
					save_test_F_CPU((char *)buffer, (float *)&h_result, maxiter * 100 + repetition_i, 1);
				}

				fprintf(stderr, "diff = %f\n", diff_average);

	//			saveGPUrealtxt_C(dev_solution, "/output/solution.txt", N * N);

				cudacall(cudaFree((bool *)dev_mask));
				cudacall(cudaFree((cuComplex *)dev_solution));
				cudacall(cudaFree((cuComplex *)dev_gamma_array));
				cudacall(cudaFree((cuComplex *)dev_analytical_solution));
				cufftcall(cufftDestroy(plan));
				cusolverDnDestroy(cusolverH);
				free((timespec *)h_computation_times);
				cublascall(cublasDestroy_v2(handle));
                                cudacall(cudaFree((float *)dev_actual_residual));
			}
		}
	}

	free(n_timestamps_array);
}
