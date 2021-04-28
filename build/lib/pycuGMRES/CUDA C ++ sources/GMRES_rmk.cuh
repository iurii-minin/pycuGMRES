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

	Fast_GMRES_with_CUDA(	
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
