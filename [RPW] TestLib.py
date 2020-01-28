from pycuGMRES import *

pow_given = 10
N = 1 << pow_given
Nsqrd = 1 << (pow_given << 1)

wavelength_per_domain = 6
k = 2 * 3.14 / ( N / wavelength_per_domain )

file_an_sol = '/gpfs/gpfs0/i.minin/input/analytical_solution_' + \
               str (N) + '.txt'

n_timestamps_array = get_n_timestamps_array_improved(max_maxiter = 50)


visible_device = 0
h_visible_device = c_uint(visible_device)
pycuSetDevice(h_visible_device)

maxiter = 30
tolerance = 0.00001

cylinder_mask = get_cylinder_mask(N)                                   # bool *h_mask

h_mask = (c_bool * Nsqrd)()
py_to_ctype(cylinder_mask, h_mask)
# memmove(h_mask, cylinder_mask.ctypes.data, cylinder_mask.nbytes)
dev_mask = pycumalloc(Nsqrd, c_size_t(sizeof(c_bool)))
pycuhost2gpu(h_mask, dev_mask, Nsqrd, c_size_t(sizeof(c_bool)))
dev_mask = cast(dev_mask, POINTER(c_bool))

dev_solution = pycumalloc(Nsqrd, c_size_t(sizeof(c_complex)))          # cuComplex *dev_solution
dev_solution = cast(dev_solution, POINTER(c_complex))
h_for_gradient = c_bool(False)                                         # const bool for_gradient
h_index_of_max = c_uint(0)                                             # const unsigned int h_index_of_max
h_tolerance = c_float(tolerance)                                       # const float tolerance
h_GMRES_n = (c_uint * 1)(0)                                            # unsigned int *GMRES_n
dev_actual_residual = pycumalloc(maxiter+1, c_size_t(sizeof(c_float))) # float *dev_actual_residual
dev_actual_residual = cast(dev_actual_residual, POINTER(c_float))
h_res_vs_tol_p = c_bool(True)                                          # bool *h_res_vs_tol_p
h_N = c_uint(N)                                                        # const unsigned int N    
pycuInitSolution(dev_solution, h_N)

h_gamma_array = (c_complex * (2 * N - 1) ** 2 )()                      # cuComplex *dev_gamma_array
gamma_array = get_gamma_array(k, N)
py_to_ctype(gamma_array, h_gamma_array)
memmove(h_gamma_array, gamma_array.ctypes.data, gamma_array.nbytes)
dev_gamma_array = pycumalloc((2 * N - 1) ** 2, c_size_t(sizeof(c_complex)))
pycuhost2gpu(h_gamma_array, dev_gamma_array, (2 * N - 1) ** 2, c_size_t(sizeof(c_complex)))
dev_gamma_array = cast(dev_gamma_array, POINTER(c_complex))
h_plan = pycuGetPlan(h_N)                                              # const cufftHandle plan
h_handle_p = pycuHandleBlas()                                          # cublasHandle_t handle
h_cusolverH_p = pycuHandleSolverDn()                                   # cusolverDnHandle_t *cusolverH_p

h_maxiter = c_uint(maxiter)                                            # unsigned int maxiter

dev_subs = (c_devSubsidiary * 1)()                                     # dev_subsidiary *dev_subs
pycuGetSubsidiary(dev_subs, h_N, h_maxiter)
n_timestamps = n_timestamps_array[maxiter]                             # timespec *computation_times
h_computation_times = (c_timespec * n_timestamps)()

pycuSetPointerMode(h_handle_p, CUBLAS_POINTER_MODE_DEVICE())

time_c = time()

pycuGMRES(
    dev_mask,
    dev_solution,
    h_for_gradient,
    h_index_of_max,
    h_maxiter,
    h_tolerance,
    h_GMRES_n,
    dev_actual_residual,
    h_res_vs_tol_p,
    h_N,
    dev_gamma_array,
    h_plan,
    h_handle_p,
    h_cusolverH_p,
    dev_subs,
    h_computation_times
         )

time_c = time() - time_c

print("time_c for GMRES = ", time_c)

h_analytical_solution = (c_complex * Nsqrd )()    # cuComplex *dev_analytical_solution
analytical_solution = get_complex_array(file_an_sol)
py_to_ctype(analytical_solution, h_analytical_solution)
dev_analytical_solution = pycumalloc(Nsqrd, c_size_t(sizeof(c_complex)))
pycuhost2gpu(h_analytical_solution, dev_analytical_solution, Nsqrd, c_size_t(sizeof(c_complex)))
dev_analytical_solution = cast(dev_analytical_solution, POINTER(c_complex))

pycuSetPointerMode(h_handle_p, CUBLAS_POINTER_MODE_HOST())
pycuRelErr(
            dev_solution, 
            dev_analytical_solution,
            h_N,
            h_handle_p
          )

pycuDestroyPlan(h_plan)
pycuDestroyBlas(h_handle_p)
pycuDestroySolverDn(h_cusolverH_p)
pycuFree(dev_gamma_array)
pycuFree(dev_mask)
pycuFree(dev_actual_residual)
pycuDestroySubsidiary(dev_subs)
pycuFree(dev_analytical_solution)

print("Total time = ", np.sum(get_nano_time(h_computation_times)/1e9))

