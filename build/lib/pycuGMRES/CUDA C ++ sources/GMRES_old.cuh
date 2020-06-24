void cuSolve_LES(	cuComplex *d_A,
			cuComplex *d_B,
			const int m,
			cusolverDnHandle_t cusolverH,
			time_t *h_computation_times,
			unsigned int *clock_i_p);


void pycuGMRESold(
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
		    timespec *h_computation_times_ts)
{
	cusolverDnHandle_t cusolverH = *cusolverH_p;
  time_t *h_computation_times = (time_t *)malloc(300000 * sizeof(time_t));

	unsigned int clock_i = 0;

	h_computation_times[clock_i++] = clock(); //_0_

	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 blocks_M(THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_M);
	dim3 threads(Q, Q);
	dim3 blocksize(512);
	dim3 gridsize(N * N / blocksize.x);

	bool h_res_vs_tol = false;

	bool *dev_res_vs_tol			 = NULL;
	
	cuComplex *dev_residual_vec		 = NULL;
	cuComplex *dev_orthogonal_basis		 = NULL;
	cuComplex *dev_HH			 = NULL;
	cuComplex *dev_Jtotal			 = NULL;
	cuComplex *dev_H_			 = NULL;
	cuComplex *dev_Htemp			 = NULL;
	cuComplex *dev_cc			 = NULL;
	cuComplex *dev_Givens_rotation		 = NULL;
	cuComplex *dev_alpha			 = NULL;
	cuComplex *dev_beta			 = NULL;
	cuComplex *dev_matmul_out_extended	 = NULL;
	cuComplex *dev_temporary		 = NULL;
	cuComplex *dev_w			 = NULL;
	cuComplex *dev_resized			 = NULL;

	float *dev_Hjk_norm_part		 = NULL;
	float *dev_scal_mul			 = NULL;

	unsigned int GMRES_i = 0;

	cudacall(cudaMalloc((void**)&dev_temporary, 					sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_alpha, 					sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_beta, 						sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_residual_vec, 				N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_orthogonal_basis, 	(maxiter + 1) * N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_Hjk_norm_part, 				    sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_scal_mul, 					    sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_res_vs_tol, 					     sizeof(bool)));

	cudacall(cudaMalloc((void**)&dev_cc, 			  	  (maxiter + 1) * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_matmul_out_extended, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_H_,       				      2 * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_w , 					  N * N * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_Htemp, 				      2 * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_Givens_rotation, 			      4 * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_Jtotal, 				      4 * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_HH, 	          (maxiter + 1) * (maxiter + 1) * sizeof(cuComplex)));


	h_computation_times[clock_i++] = clock(); //_1_	//Initialization
//========================================= BEGIN: get_residual_vector =======================================================


	if (for_gradient)
	{
		G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
			(cuComplex *)dev_solution,
			(cuComplex *)dev_matmul_out_extended,
			(cufftHandle)plan, N);

		h_computation_times[clock_i++] = clock(); //_2_

		_2D_to_1D_compared_kernel <<< blocks, threads >>> (	(bool *)dev_mask,
									(cuComplex *)dev_solution,
									(cuComplex*)dev_matmul_out_extended,
									(cuComplex*)dev_residual_vec,
									h_index_of_max, N);
	}
	else
	{
		G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
			(bool *)dev_mask,
			(cuComplex *)dev_solution,
			(cuComplex *)dev_matmul_out_extended,
			(cufftHandle)plan, N);

		h_computation_times[clock_i++] = clock(); //_2_

		_2D_to_1D_compared_kernel <<< blocks, threads >>> (	(cuComplex *)dev_solution,
									(cuComplex*)dev_matmul_out_extended,
									(cuComplex*)dev_residual_vec, N);
	}
	cudacheckSYN();

	h_computation_times[clock_i++] = clock(); //_3_
//========================================== END: get_residual_vector =========================================================
	cublascall(cublasScnrm2(        (cublasHandle_t) *handle_p,
					N * N,
		                        (const cuComplex *)dev_residual_vec, 1,
					(float  *)dev_actual_residual));
	cudacheckSYN();

	h_computation_times[clock_i++] = clock(); //_4_
//============================================= Begin: Condition to iterate ==========================================================
	residual_vs_tolerance_kernel <<< 1, 1 >>> (	(float *)dev_actual_residual,
							(bool *)dev_res_vs_tol,
							tolerance);
	cudacheckSYN();

	cudacall(cudaMemcpy(&h_res_vs_tol, dev_res_vs_tol, sizeof(bool), cudaMemcpyDeviceToHost));

	h_computation_times[clock_i ++] = clock(); //_5_
//=============================================== End: Condition to iterate ===========================================================
//============================================BEGIN:residual_normalization_kernel=======================================================
	residual_normalization_kernel <<< gridsize, blocksize >>> (	(cuComplex *)dev_residual_vec,
									(float *)dev_actual_residual,
									(cuComplex *)dev_orthogonal_basis);
	cudacheckSYN();

	set_alpha_beta <<< 4, 1 >>> ((cuComplex *)dev_alpha, (cuComplex *)dev_beta);
	//don't synchronize

	h_computation_times[clock_i++] = clock(); //_6_
//============================================= END:residual_normalization_kernel ==================================================
	if (h_res_vs_tol)
	{
		h_computation_times[clock_i ++] = clock(); //_7_

		if (for_gradient)
		{
			G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
					(cuComplex *)dev_orthogonal_basis,
					(cuComplex *)dev_matmul_out_extended,
					(cufftHandle) plan, N);

			h_computation_times[clock_i ++] = clock(); //_8_

			_2D_to_1D_kernel <<< blocks, threads >>> (	(bool *)dev_mask,
									(cuComplex*)dev_orthogonal_basis,
									(cuComplex *)dev_matmul_out_extended,
									(cuComplex *)dev_w, N);
		}
		else
		{
			G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
					(bool *)dev_mask,
					(cuComplex *)dev_orthogonal_basis,
					(cuComplex *)dev_matmul_out_extended,
					(cufftHandle) plan, N);

			h_computation_times[clock_i ++] = clock(); //_8_

			_2D_to_1D_kernel <<< blocks, threads >>> (	(cuComplex*)dev_orthogonal_basis,
									(cuComplex *)dev_matmul_out_extended,
									(cuComplex *)dev_w, N);
		}
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_9_

		cublascall(cublasCdotc(		(cublasHandle_t) *handle_p, N * N,
						(const cuComplex *)dev_orthogonal_basis, 1,
						(const cuComplex *)dev_w, 1,
						(cuComplex *)dev_H_));
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_10_

		set_inverse_kernel <<< 1, 2 >>> ((const cuComplex *)dev_H_, (cuComplex *)dev_temporary);
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_!1_

		cublascall(cublasCaxpy(	(cublasHandle_t) *handle_p, N * N,
					(const cuComplex *)dev_temporary, 
					(const cuComplex *)dev_orthogonal_basis, 1, 
					(cuComplex *)dev_w, 1));
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_11_

		cublascall(cublasScnrm2(        (cublasHandle_t) *handle_p,
						N * N,
				                (const cuComplex *)dev_w, 1,
						(float  *)dev_Hjk_norm_part));
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_12_
	//============================================== BEGIN: Fill Orthogonal Basis matrix ============================================
		get_complex_divided <<< 3, 1 >>> (	(const float *)dev_Hjk_norm_part,
							(cuComplex *)dev_H_ + 1,
							(float *)dev_scal_mul);
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_13_

		cublascall(cublasCsscal(	(cublasHandle_t) *handle_p, N * N,
				            	(const float           *)dev_scal_mul,
				            	(cuComplex       *)dev_w, 1));
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_14_

		cublascall(cublasCcopy(		(cublasHandle_t) *handle_p, N * N,
				           	(const cuComplex       *)dev_w, 1,
				           	(cuComplex             *)dev_orthogonal_basis + N * N, 1));
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_!2_
	//============================================== END: Orthogonal Basis matrix  ==================================================
		cublascall(cublasCcopy(		(cublasHandle_t) *handle_p, 2,
				           	(const cuComplex       *)dev_H_, 1,
				           	(cuComplex             *)dev_Htemp, 1));
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_!3_
	//============================================= BEGIN: Create Jtotal_Matrix ========================================
		create_Givens_rotation_matrix_kernel <<< dim3(2, 2), dim3(1, 1) >>> ((cuComplex *)dev_Givens_rotation, (cuComplex *)dev_Htemp);
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_15_
	//=============================================== END: Create Jtotal_Matrix,  ========================================
	//============================================= BEGIN: Create Givens_Rotation_Matrix ========================================
		cublascall(cublasCcopy(		(cublasHandle_t) *handle_p, 4,
				           	(const cuComplex       *)dev_Givens_rotation, 1,
				           	(cuComplex             *)dev_Jtotal, 1));
		cudacheckSYN();

		h_computation_times[clock_i ++] = clock(); //_16_
	//=============================================== END: Create Givens_Rotation_Matrix,  ========================================
	//===================================================== BEGIN: Update residual ======================================================
		next_residual_kernel <<< 1, 1 >>> ((cuComplex *)dev_Jtotal + 2, (float *)dev_actual_residual, (float *)dev_actual_residual + 1);
		cudacheckSYN();

		residual_vs_tolerance_kernel <<< 1, 1 >>> (	(float *)dev_actual_residual + 1,
								(bool *)dev_res_vs_tol,
								tolerance);
		cudacheckSYN();
		cudacall(cudaMemcpy(&h_res_vs_tol, dev_res_vs_tol, sizeof(bool), cudaMemcpyDeviceToHost));

		h_computation_times[clock_i ++] = clock(); //_17_
	//======================================================= END: Update residual ======================================================
		GMRES_i ++;

		for(GMRES_i = 1; (h_res_vs_tol && (GMRES_i < maxiter)); GMRES_i ++) //
		{
			h_computation_times[clock_i ++] = clock(); //_18_

			get_resized(	(cuComplex **)&dev_H_,
					dim3(GMRES_i + 2, GMRES_i + 1),
					dim3(1, 1),
					(unsigned int)GMRES_i + 1,
					(unsigned int)GMRES_i,
					(unsigned int)GMRES_i + 2,
					(unsigned int)GMRES_i + 1);

			h_computation_times[clock_i ++] = clock(); //_!4_


			if (for_gradient)
			{
				G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
						(cuComplex *)dev_orthogonal_basis + GMRES_i * N * N,
						(cuComplex *)dev_matmul_out_extended,
						(cufftHandle) plan, N);

				h_computation_times[clock_i ++] = clock(); //_19_

				_2D_to_1D_kernel <<< blocks, threads >>> (	(bool *)dev_mask,
										(cuComplex*)dev_orthogonal_basis + GMRES_i * N * N,
										(cuComplex *)dev_matmul_out_extended,
										(cuComplex *)dev_w, N);
			}
			else
			{
				G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
						(bool *)dev_mask,
						(cuComplex *)dev_orthogonal_basis + GMRES_i * N * N,
						(cuComplex *)dev_matmul_out_extended,
						(cufftHandle) plan, N);

				h_computation_times[clock_i ++] = clock(); //_19_

				_2D_to_1D_kernel <<< blocks, threads >>> (	(cuComplex*)dev_orthogonal_basis + GMRES_i * N * N,
										(cuComplex *)dev_matmul_out_extended,
										(cuComplex *)dev_w, N);
			}
			cudacheckSYN();

			h_computation_times[clock_i ++] = clock(); //_20_

			for(unsigned int j = 0; j < GMRES_i + 1; j++)
			{
				cublascall(cublasCdotc(	(cublasHandle_t) *handle_p, N * N,
							(const cuComplex *)dev_orthogonal_basis + j * N * N, 1,
							(const cuComplex *)dev_w, 1,
							(cuComplex *)dev_H_ + j * (GMRES_i + 1) + GMRES_i));
				cudacheckSYN();

				h_computation_times[clock_i ++] = clock(); //_21_

				set_inverse_kernel <<< 1, 2 >>> ((const cuComplex *)dev_H_ + j * (GMRES_i + 1) + GMRES_i, (cuComplex *)dev_temporary);
				cudacheckSYN();

				h_computation_times[clock_i ++] = clock(); //_!5_

				cublascall(cublasCaxpy(	(cublasHandle_t) *handle_p, N * N,
							(const cuComplex *)dev_temporary, 
							(const cuComplex *)dev_orthogonal_basis + j * N * N, 1, 
							(cuComplex *)dev_w, 1));
				cudacheckSYN();

				h_computation_times[clock_i ++] = clock(); //_22_
			}

			cublascall(cublasScnrm2(        (cublasHandle_t) *handle_p,
							N * N,
						        (const cuComplex *)dev_w, 1,
							(float  *)dev_Hjk_norm_part));
			cudacheckSYN();

			h_computation_times[clock_i ++] = clock(); //_23_
		//============================================== BEGIN: Fill Orthogonal Basis m.============================================
			get_complex_divided <<< 3, 1 >>> (	(const float *)dev_Hjk_norm_part,
								(cuComplex *)dev_H_+(GMRES_i + 1) * (GMRES_i + 1) + GMRES_i,
								(float *)dev_scal_mul);
			cudacheckSYN();
			h_computation_times[clock_i ++] = clock(); //_24_

			cublascall(cublasCsscal(		(cublasHandle_t) *handle_p, N * N,
								(const float           *)dev_scal_mul,
								(cuComplex       *)dev_w, 1));
			cudacheckSYN();
			h_computation_times[clock_i ++] = clock(); //_25_

			cublascall(cublasCcopy(		(cublasHandle_t) *handle_p, N * N,
						   	(const cuComplex       *)dev_w, 1,
						   	(cuComplex             *)dev_orthogonal_basis + (GMRES_i + 1) * N * N, 1));
			cudacheckSYN();

			h_computation_times[clock_i ++] = clock(); //_!6_
		//===============================    END: Fill Orthogonal Basis m.  ===========================================
		//============================================== Begin: Least Squares Step =========================================================
		//========================================== BEGIN:(Jtotal)resize_kernel ==========================================
			cudacall(cudaMalloc((void**)&dev_resized, (GMRES_i + 2) * (GMRES_i + 2) * sizeof(cuComplex)));
	
			Jtotal_resize_kernel <<< dim3(GMRES_i + 2, GMRES_i + 2), dim3(1, 1) >>> ((cuComplex *)dev_Jtotal, GMRES_i + 1, (cuComplex *)dev_resized);
			cudacheckSYN();

			cudacall(cudaFree((cuComplex *)dev_Jtotal));
			dev_Jtotal = dev_resized;
			dev_resized = NULL;

			h_computation_times[clock_i ++] = clock(); //_!8_
		//====================================== END: (Jtotal) resize_kernel ============================================
		//================================ BEGIN: MATMUL (H_temp=Jtotal * H) ==============================================
			cudacall(cudaFree((cuComplex *)dev_Htemp));
			cudacall(cudaMalloc((void**)&dev_Htemp, (GMRES_i + 2) * (GMRES_i + 1) * sizeof(cuComplex)));

			h_computation_times[clock_i ++] = clock(); //_!9_

			usual_MatMul_CUDA_Device_Mode((cublasHandle_t *)handle_p, (cuComplex *)dev_Jtotal, (cuComplex *)dev_H_, (cuComplex *)dev_Htemp, (unsigned int)(GMRES_i + 2), (unsigned int)(GMRES_i + 2), (unsigned int)(GMRES_i + 1), (const cuComplex *)dev_alpha, (const cuComplex *)dev_beta);
			cudacheckSYN();

			h_computation_times[clock_i ++] = clock(); //_26_
		//================================== END: MATMUL (H_temp=Jtotal * H) ===============================================
		//================================================ END: Least Squares Step =========================================================
		//============================================= BEGIN: Create Givens_Rotation_Matrix ========================================
			cudacall(cudaFree((cuComplex *)dev_Givens_rotation));
			cudacall(cudaMalloc((void**)&dev_Givens_rotation, (GMRES_i + 2) * (GMRES_i + 2) * sizeof(cuComplex)));

			h_computation_times[clock_i ++] = clock(); //_!10_

			create_Givens_rotation_matrix_kernel <<< dim3(GMRES_i + 2, GMRES_i + 2), dim3(1, 1) >>> ((cuComplex *)dev_Givens_rotation, (cuComplex *)dev_Htemp);
			cudacheckSYN();

			h_computation_times[clock_i ++] = clock(); //_27_
		//=============================================== END: Create Givens_Rotation_Matrix ========================================
		//================================================== BEGIN: Jtotal = J*Jtotal =================================================
			usual_MatMul_CUDA_Device_Mode((cublasHandle_t *)handle_p, (cuComplex *)dev_Givens_rotation, (cuComplex *)dev_Jtotal, (cuComplex *)dev_Jtotal, (unsigned int)GMRES_i + 2, (unsigned int)(GMRES_i + 2), (unsigned int)(GMRES_i + 2), (const cuComplex *)dev_alpha, (const cuComplex *)dev_beta);
			cudacheckSYN();

			h_computation_times[clock_i ++] = clock(); //_28_
		//==================================================== END: Jtotal = J*Jtotal =================================================
		//===================================================== BEGIN: Update residual ======================================================
			next_residual_kernel <<< 1, 1 >>> (	(cuComplex *)dev_Jtotal + (GMRES_i + 2) * (GMRES_i + 1),
								(float *)dev_actual_residual,
								(float *)dev_actual_residual + GMRES_i + 1);

			residual_vs_tolerance_kernel <<< 1, 1 >>> (	(float *)(dev_actual_residual + GMRES_i + 1),
									(bool *)dev_res_vs_tol, tolerance);
			cudacheckSYN();

			cudacall(cudaMemcpy(&h_res_vs_tol, dev_res_vs_tol, sizeof(bool), cudaMemcpyDeviceToHost));
			cudacheckSYN();

			h_computation_times[clock_i ++] = clock(); //_29_
		//======================================================= END: Update residual ======================================================
		}
	//================================================= BEGIN: Free after loop ==================================================================
	//=================================================== END: Free after loop ==================================================================
	//================================================== BEGIN: HH = (Jtotal*H)_resized ==========================================================
		usual_MatMul_CUDA_Device_Mode((cublasHandle_t *)handle_p, (cuComplex *)dev_Jtotal, (cuComplex *)dev_H_, (cuComplex *)dev_HH,(unsigned int)GMRES_i,(unsigned int)(GMRES_i + 1),(unsigned int)GMRES_i, (const cuComplex *)dev_alpha, (const cuComplex *)dev_beta);
		cudacheckSYN();
		
		h_computation_times[clock_i ++] = clock(); //_30_


		cudacall(cudaMalloc((void**)&dev_resized, GMRES_i * GMRES_i * sizeof(cuComplex)));

		cublascall(cublasCgeam(		(cublasHandle_t) *handle_p,
				          	CUBLAS_OP_T, CUBLAS_OP_N,
				          	GMRES_i, GMRES_i,
				          	(const cuComplex       *)dev_alpha,
				          	(const cuComplex       *)dev_HH, GMRES_i,
				          	(const cuComplex       *)dev_beta ,
				          	(const cuComplex       *)dev_HH, GMRES_i,
				          	(cuComplex       *)dev_resized, GMRES_i));

		cudacall(cudaFree(dev_HH));
		dev_HH = dev_resized;
		dev_resized = NULL;

		h_computation_times[clock_i ++] = clock(); //_!11_
	//===================================================== END: HH = (Jtotal*H)_resized ==========================================================
	//================================================= BEGIN: cc = Jtotal * norm_res_vec =========================================================
		set_cc_kernel <<< GMRES_i, 1 >>> (	(cuComplex *)dev_cc,
							(cuComplex *)dev_Jtotal,
							(float *)dev_actual_residual,
							maxiter + 1);
		cudacheckSYN();
		h_computation_times[clock_i ++] = clock(); //_31_
	//=================================================== END: cc = Jtotal * norm_res_vec =========================================================
		if (GMRES_i > 0)
		{
			if (GMRES_i < 2)
			{
				get_new_solution_kernel <<< 1, 1 >>> (	(cuComplex *)dev_cc,
									(cuComplex *)dev_HH);	
				cudacheckSYN();

				get_solution_kernel <<< gridsize, blocksize >>> (	(cuComplex *)dev_solution,
											(cuComplex *)dev_cc,
											(cuComplex *)dev_orthogonal_basis);
				cudacheckSYN();
			}
			else
			{
			//============================================ BEGIN: Find solution to the LES(cc_new) for HH*cc_new=cc ============================================
				cuSolve_LES((cuComplex *)dev_HH, (cuComplex *)dev_cc, GMRES_i, cusolverH, (time_t *)h_computation_times, (unsigned int *)&clock_i);
			//============================================ END: Find solution to the LES(cc_new) for HH*cc_new=cc ===========================================
			//============================================ BEGIN: x = x0 + V * cc ===========================================
				for(unsigned int j = 0; j < GMRES_i; j++)
				{
					add_kernel <<< gridsize, blocksize >>> ((cuComplex *)dev_solution, (cuComplex *)dev_orthogonal_basis + j * N * N, (cuComplex *)dev_cc + j);
					//cudacheckSYN();

					h_computation_times[clock_i ++] = clock(); //_34_
				}
			}
		}
	}
	*GMRES_n	 = GMRES_i;
	*h_res_vs_tol_p	 = h_res_vs_tol;
	h_computation_times[clock_i ++] = clock(); //_35_

	cudacall(cudaFree((bool *)dev_res_vs_tol));
	cudacall(cudaFree((cuComplex *)dev_residual_vec));
	cudacall(cudaFree((cuComplex *)dev_orthogonal_basis));
	cudacall(cudaFree((cuComplex *)dev_HH));
	cudacall(cudaFree((cuComplex *)dev_Jtotal));
	cudacall(cudaFree((cuComplex *)dev_H_));
	cudacall(cudaFree((cuComplex *)dev_Htemp));
	cudacall(cudaFree((cuComplex *)dev_cc));
	cudacall(cudaFree((cuComplex *)dev_Givens_rotation));
	cudacall(cudaFree((cuComplex *)dev_temporary));
	cudacall(cudaFree((cuComplex *)dev_alpha));
	cudacall(cudaFree((cuComplex *)dev_beta));
	cudacall(cudaFree((cuComplex *)dev_matmul_out_extended));
	cudacall(cudaFree((float *)dev_Hjk_norm_part));
	cudacall(cudaFree((float *)dev_scal_mul));
	cudacall(cudaFree((cuComplex *)dev_resized));

	h_computation_times[clock_i ++] = clock(); //_36_
}



void cuSolve_LES(cuComplex *d_A, cuComplex *d_B, const int m, cusolverDnHandle_t cusolverH, time_t *h_computation_times, unsigned int *clock_i_p)
{
	int *d_Ipiv = NULL; /* pivoting sequence */
	int *d_info = NULL; /* error info */
	int  lwork = 0;     /* size of workspace */
	cuComplex *d_work = NULL; /* device workspace for getrf */

	const int lda = m;
	const int ldb = m;

	int info = 0;     /* host copy of error info */

	h_computation_times[(*clock_i_p)++] = clock(); //_32_

	/* step 1: query working space of getrf */
	cudacall(cudaMalloc((void**)&d_Ipiv, sizeof(int) * m));
	cudacall(cudaMalloc((void**)&d_info, sizeof(int)));


	h_computation_times[(*clock_i_p)++] = clock(); //_!12_

	cusolvercall(cusolverDnCgetrf_bufferSize(	cusolverH,
							m,
							m,
							d_A,
							lda,
							&lwork));

	cudacall(cudaMalloc((void**)&d_work, lwork * sizeof(cuComplex)));

	/* step 2: LU factorization */
	cusolvercall(cusolverDnCgetrf(      cusolverH,
					    m,
					    m,
					    d_A,
					    lda,
					    d_work,
					    d_Ipiv,
					    d_info));
	cudacheckSYN();

	cudacall(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

	if ( 0 > info )
	{
		printf("%d-th parameter is wrong \n", -info);
		exit(1);
	}

	cusolvercall(cusolverDnCgetrs(  cusolverH,
					CUBLAS_OP_N,
					m,
					1, /* nrhs */
					d_A,
					lda,
					d_Ipiv,
					d_B,
					ldb,
					d_info));
	cudacheckSYN();
	cudacall(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

	if ( info != 0 )
	{
		printf("%d-th parameter is wrong \n", -info);
		exit(1);
	}

	h_computation_times[(*clock_i_p)++] = clock(); //_33_

	if (d_Ipiv ) cudaFree(d_Ipiv);
	if (d_info ) cudaFree(d_info);
	if (d_work ) cudaFree(d_work);
	h_computation_times[(*clock_i_p)++] = clock(); //_!13_
}
