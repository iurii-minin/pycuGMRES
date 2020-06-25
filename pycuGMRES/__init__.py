from ctypes import *#c_void_p, c_size_t
import os
import numpy as np
import scipy.special
import copy
import matplotlib.pyplot as plt
from time import time

import pkg_resources



class c_complex(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float)]

class c_timespec(Structure):
    _fields_ = [("tv_sec",  c_ulonglong),
                ("tv_nsec", c_ulonglong)]

class c_devSubsidiary(Structure):
    _fields_ = [("dev_orthogonal_basis",  POINTER(c_complex)),
                ("dev_info", POINTER(c_int))]


def CUBLAS_POINTER_MODE_HOST():
    return c_uint(0)

def CUBLAS_POINTER_MODE_DEVICE():
    return c_uint(1)


def get_function(fun_description, path_to_so):
    so_CDLL = CDLL(path_to_so)
    prefix = "objdump -T " + '"' + path_to_so + '"' + " | tr ' ' '\n' | grep "
    string_fun_so = os.popen(prefix + fun_description).read()
    string_fun_so = string_fun_so.split("\n")[-2] #choose last search result
    return getattr(so_CDLL, string_fun_so)

def get_cylinder_mask(m):    
    cylinder_mask = np.zeros((m, m)).astype('bool')
    x, y = np.mgrid[0:m, 0:m]
    cylinder_mask[(x- m/3)**2 + (y - m / 2)**2 <= (m/ 6) ** 2 ] = True
    return cylinder_mask.reshape(-1)

def get_greenfun(r,k):
    return (1j/4)*scipy.special.hankel1(0, k*r)

def get_green_matrix(k,size):
    j,i = np.mgrid[:size, :size]
    ij_block = np.sqrt((i-1/2)**2+j**2)
    green_mat = get_greenfun(ij_block, k)
    return green_mat

def get_toeplitz_mat(ij_block):
    ij_block = copy.deepcopy(ij_block)
    T1 = np.hstack((ij_block,ij_block[:,:0:-1]))
    T2 = np.hstack((ij_block[:0:-1,:],ij_block[:0:-1,:0:-1]))
    T = np.vstack((T1,T2))
    return np.asarray(T, np.complex64)

def get_gamma_array(k, size):
    G_block = get_green_matrix(k, size)
    return get_toeplitz_mat(G_block).reshape(-1)

def py_to_ctype(in_py, ctype_out):
    memmove(ctype_out, in_py.ctypes.data, in_py.nbytes)
        
def get_complex_array(filename, transpose = False):
    
    N_big = 0
    mynumbers = []
    with open(filename) as f:
        for line in f:
            N_big += 1
            mynumbers.append([float(n) for n in line.strip().split(' ')])

    N_small = int(np.sqrt(N_big))
    complex_array = np.zeros(N_big, dtype = np.complex64)
    
    if transpose:
        j = 0
        i = 0

        for pair in mynumbers:
            try:
                index = i * N_small + j
                complex_array[index] = pair[0] + 1j * pair[1]
                if i < N_small - 1:
                    i += 1
                else:
                    i = 0
                    j += 1            

            except IndexError:
                print("A line in the file doesn't have enough entries.")
    else:
        index = 0

        for pair in mynumbers:
            try:
                complex_array[index] = pair[0] + 1j * pair[1]
                index += 1

            except IndexError:
                print("A line in the file doesn't have enough entries.")
    return complex_array

def visualize(data, wavelength_per_domain = 10, title = "",cmap='magma', title_max = True, savefig = False, show_cbar = True, iteration = None):
    plt.figure(figsize=(18, 18))
    if title_max: 
        plt.title(title+"Maximal value = %.5f" % np.max(data), fontsize = 45)
    else:
        plt.title(title, fontsize = 45)
        
    neg = plt.imshow(data, cmap=cmap, interpolation='none')
    if show_cbar:
        cbar =  plt.colorbar(neg)    
        cbar.ax.tick_params(labelsize = 35)
    plt.xticks(np.arange(wavelength_per_domain+1)*data.shape[0]/wavelength_per_domain, np.arange(wavelength_per_domain+1), fontsize = 35)
    plt.yticks(np.arange(wavelength_per_domain+1)*data.shape[0]/wavelength_per_domain, np.arange(wavelength_per_domain+1), fontsize = 35)
    plt.xlabel("Ox, wavelength", fontsize = 35)
    plt.ylabel("Oy, wavelength", fontsize = 35)
    if savefig:
        if iteration is None:
            plt.savefig('Python_images/'+title + '.%.5f.png' % time.time(), bbox_inches = 'tight')
        else:        
            plt.savefig('Python_images/'+title+'.%i.png' % iteration, bbox_inches = 'tight')
    
def get_n_timestamps_val(maxiter = 100): #Comparables/new
    n_timestamps  = 1; #short_indexed_text_array = []
    n_timestamps += 1; #short_indexed_text_array.append("Initialization (malloc)") #_1_ !_
    n_timestamps += 1; #short_indexed_text_array.append("G_x_fft_matvec for A*x0") #_2_ !_
    n_timestamps += 1; #short_indexed_text_array.append("2D_to_1D for A*x0-x0") #_3_
    n_timestamps += 1; #short_indexed_text_array.append("Norm(residual_vec)") #_4_
    n_timestamps += 1; #short_indexed_text_array.append("Condition to iterate") #_5_ !_
    n_timestamps += 1; #short_indexed_text_array.append("Residual_normalization & set_a,b") #_6_
    
    GMRES_i = 0
    
    if True:
        n_timestamps += 1; #short_indexed_text_array.append("Memset(H, 0)") #_7_ !_
        n_timestamps += 1; #short_indexed_text_array.append("G_x_fft_matvec for w=A*v iteration(" + str(GMRES_i) + ")") #_8_
        n_timestamps += 1; #short_indexed_text_array.append("2D_to_1D for w=A*v iteration(" + str(GMRES_i) + ")") #_9_
        n_timestamps += 1; #short_indexed_text_array.append("H_jk = (V_j, w) iteration(" + str(GMRES_i) + ")") #_10_
        n_timestamps += 1; #short_indexed_text_array.append("w = w - H*v iteration(" + str(GMRES_i) + ")") #_11_ !_    
        n_timestamps += 1; #short_indexed_text_array.append("H_jj+1 = norm(w) iteration(" + str(GMRES_i) + ")") #_12_    
        n_timestamps += 1; #short_indexed_text_array.append("1/H_jj+1 iteration(" + str(GMRES_i) + ")") #_13_    
        n_timestamps += 1; #short_indexed_text_array.append("w = w/H_jj+1 iteration(" + str(GMRES_i) + ")") #_14_
        n_timestamps += 1; #short_indexed_text_array.append("Set(J) iteration(" + str(GMRES_i) + ")") #_15_ !_
        n_timestamps += 1; #short_indexed_text_array.append("Set(Jtotal) iteration(" + str(GMRES_i) + ")") #_16_ !_
        n_timestamps += 1; #short_indexed_text_array.append("Update residual iteration(" + str(GMRES_i) + ")") #_17_ !_
        
        for GMRES_i in range(1, maxiter):  
            n_timestamps += 1; #short_indexed_text_array.append("Condition_check iteration(" + str(GMRES_i) + ")") #_18_
            n_timestamps += 1; #short_indexed_text_array.append("G_x_fft_matvec for w=A*v iteration(" + str(GMRES_i) + ")") #_19_        
            n_timestamps += 1; #short_indexed_text_array.append("2D_to_1D for w=A*v iteration(" + str(GMRES_i) + ")") #_20_     
                
            for j in range(GMRES_i + 1):
                n_timestamps += 1; #short_indexed_text_array.append("H_jk = (V_j, w) iteration(" + str(GMRES_i) + ", j = " + str(j) + ")") #_21_
                n_timestamps += 1; #short_indexed_text_array.append("w = w - H_jk * V_j iteration(" + str(GMRES_i) + ", j = " + str(j) + ")") #_22_  
                   
                
            n_timestamps += 1; #short_indexed_text_array.append("H_jj+1 = norm(w) iteration(" + str(GMRES_i) + ")") #_23_
            n_timestamps += 1; #short_indexed_text_array.append("1/H_jj+1 iteration(" + str(GMRES_i) + ")") #_24_
            n_timestamps += 1; #short_indexed_text_array.append("w = w/H_jj+1 iteration(" + str(GMRES_i) + ")") #_25_    
            n_timestamps += 1; #short_indexed_text_array.append("H_temp=Jtotal * H iteration(" + str(GMRES_i) + ")") #_26_
            n_timestamps += 1; #short_indexed_text_array.append("Set(J) iteration(" + str(GMRES_i) + ")") #_27_ !_
            n_timestamps += 1; #short_indexed_text_array.append("Jtotal = J*Jtotal iteration(" + str(GMRES_i) + ")") #_28_
            n_timestamps += 1; #short_indexed_text_array.append("Update residual iteration(" + str(GMRES_i) + ")") #_29_ !_
            
    GMRES_i += 1
    n_timestamps += 1; #short_indexed_text_array.append("HH = Jtotal * H") #_30_
    n_timestamps += 1; #short_indexed_text_array.append("cc <- Jtotal") #_31_
    n_timestamps += 1; #short_indexed_text_array.append("Initialize_small_LES(HH, cc)") #_32_
    n_timestamps += 1; #short_indexed_text_array.append("Process_small_LES(HH, cc)") #_33_
    
    for j in range(GMRES_i):        
        n_timestamps += 1; #short_indexed_text_array.append("Add iteration(j = " + str(j) + ")") #_34_
        
    n_timestamps += 1; #short_indexed_text_array.append("set(Output_p)") #_35_
    
    return n_timestamps;

def get_n_timestamps_array(max_maxiter = 50):
    array = []
    for maxiter in range(max_maxiter):
        array.append(get_n_timestamps_val(maxiter))
    return array

def get_nano_time(h_computation_times):
    computation_times = np.asarray(h_computation_times)   
    K = len(computation_times)
    diff_sec_times = computation_times[1:K]['tv_sec'] - computation_times[0:K - 1]['tv_sec']
    diff_nsec_times = computation_times[1:K]['tv_nsec'] - computation_times[0:K - 1]['tv_nsec']
    diff_nano_times = 1e9 * diff_sec_times + diff_nsec_times
    return diff_nano_times

FOLDERGMRESdir = pkg_resources.resource_filename('pycuGMRES', '')

#print(FOLDERGMRESdir)
prefix = FOLDERGMRESdir + '/Shared object generating/'
print(os.popen('bash '+ '"' + prefix + 'compile_cpp.sh' + '" "' + FOLDERGMRESdir + '"').read())
path_to_so = prefix + 'cuGMRES.so'

pycumalloc = get_function('pycumalloc', path_to_so)
pycumalloc.argtypes = [c_uint, c_size_t]
pycumalloc.restype = c_void_p

pycuhost2gpu = get_function('pycuhost2gpu', path_to_so)
pycuhost2gpu.argtypes = [c_void_p, c_void_p, c_uint, c_size_t]

pycugpu2host = get_function('pycugpu2host', path_to_so)
pycugpu2host.argtypes = [c_void_p, c_void_p, c_uint, c_size_t]

pycuInitSolution = get_function('pycuInitSolution', path_to_so)
pycuInitSolution.argtypes = [POINTER(c_complex), c_uint]

pycuSetDevice = get_function('pycuSetDevice', path_to_so)
pycuSetDevice.argtype = c_uint

pycuGetPlan = get_function('pycuGetPlan', path_to_so)
pycuGetPlan.argtype = c_uint
pycuGetPlan.restype = c_uint

pycuGetSubsidiary = get_function('pycuGetSubsidiary', path_to_so)
pycuGetSubsidiary.argtypes = [POINTER(c_devSubsidiary), c_uint, c_uint]
pycuGetSubsidiary.restype = c_char_p

pycuHandleBlas = get_function('pycuHandleBlas', path_to_so)
pycuHandleBlas.restype = POINTER(c_longlong)

pycuHandleSolverDn = get_function('pycuHandleSolverDn', path_to_so)
pycuHandleSolverDn.restype = POINTER(c_longlong)

pycuGetGamma = get_function('pycuGetGamma', path_to_so)
pycuGetGamma.argtypes = [POINTER(c_complex), c_uint, c_uint]
pycuGetGamma.restype = POINTER(c_complex)

pycuDestroyPlan = get_function('pycuDestroyPlan', path_to_so)
pycuDestroyPlan.argtype = c_uint

pycuDestroyBlas = get_function('pycuDestroyBlas', path_to_so)
pycuDestroyBlas.argtype = POINTER(c_uint)

pycuDestroySolverDn = get_function('pycuDestroySolverDn', path_to_so)
pycuDestroySolverDn.argtype = POINTER(c_uint)

pycuFree = get_function('pycuFree', path_to_so)
pycuFree.argtype = c_void_p

pyFree = get_function('pyFree', path_to_so)
pyFree.argtype = c_void_p

pycuDestroySubsidiary = get_function('pycuDestroySubsidiary', path_to_so)
pycuDestroySubsidiary.argtype = POINTER(c_devSubsidiary)

pycuSetPointerMode = get_function('pycuSetPointerMode', path_to_so)
pycuSetPointerMode.argtypes = [POINTER(c_longlong), c_uint]

pycuRelErr = get_function('pycuRelErr', path_to_so)
pycuRelErr.argtypes = [
                              POINTER(c_complex),
                              POINTER(c_complex),
                              c_uint,
                              POINTER(c_longlong)
                      ]
pycuRelErr.restype = c_float

pycuDeviceReset = get_function('pycuDeviceReset', path_to_so)


pycuTestGMRES = get_function('pycuTestGMRES', path_to_so)

pycuFFTC2C = get_function('pycuFFTC2C', path_to_so)
pycuFFTC2C.argtypes = [
                              POINTER(c_complex), # cuComplex *dev_input
                              POINTER(c_complex), # cuComplex *dev_output
	                      c_uint              # const cufftHandle plan
		      ]

pycuGMRESimproved = get_function('pycuGMRESimproved', path_to_so)
pycuGMRESimproved.argtypes = [
                      POINTER(c_bool),           # bool *dev_mask
                      POINTER(c_complex),        # cuComplex *dev_solution
                      c_bool,                    # const bool for_gradient
                      c_uint,                    # const unsigned int h_index_of_max
                      c_uint,                    # unsigned int maxiter
                      c_float,                   # const float tolerance
                      POINTER(c_uint),           # unsigned int *GMRES_n
                      POINTER(c_float),          # float *dev_actual_residual
                      POINTER(c_bool),           # bool *h_res_vs_tol_p
                      c_uint,                    # const unsigned int N
                      POINTER(c_complex),        # cuComplex *dev_gamma_array
                      c_uint,                    # const cufftHandle plan
                      POINTER(c_longlong),       # cublasHandle_t *handle_p
                      POINTER(c_longlong),       # cusolverDnHandle_t *cusolverH_p
                      POINTER(c_devSubsidiary),  # dev_subsidiary *dev_subs
                      POINTER(c_timespec)        # timespec *computation_times
                                           ]

pycuGMRESold = get_function('pycuGMRESold', path_to_so)
pycuGMRESold.argtypes = [
                      POINTER(c_bool),           # bool *dev_mask
                      POINTER(c_complex),        # cuComplex *dev_solution
                      c_bool,                    # const bool for_gradient
                      c_uint,                    # const unsigned int h_index_of_max
                      c_uint,                    # unsigned int maxiter
                      c_float,                   # const float tolerance
                      POINTER(c_uint),           # unsigned int *GMRES_n
                      POINTER(c_float),          # float *dev_actual_residual
                      POINTER(c_bool),           # bool *h_res_vs_tol_p
                      c_uint,                    # const unsigned int N
                      POINTER(c_complex),        # cuComplex *dev_gamma_array
                      c_uint,                    # const cufftHandle plan
                      POINTER(c_longlong),       # cublasHandle_t *handle_p
                      POINTER(c_longlong),       # cusolverDnHandle_t *cusolverH_p
                      POINTER(c_devSubsidiary),  # dev_subsidiary *dev_subs
                      POINTER(c_timespec)        # timespec *computation_times
                                           ]

pycuGxFFTmatvec_grad = get_function('pycuGxFFTmatvec_grad', path_to_so)
pycuGxFFTmatvec_grad.argtypes = [	
			POINTER(c_complex), # cuComplex *dev_gamma_array,
			POINTER(c_complex), # cuComplex *dev_solution,
			POINTER(c_complex), # cuComplex *dev_matmul_out_extended,
			c_uint,             # cufftHandle plan,
			c_uint              # const unsigned int N
						]

pycu2Dto1Dgrad = get_function('pycu2Dto1Dgrad', path_to_so)
pycu2Dto1Dgrad.argtypes = [	
			POINTER(c_complex),	# cuComplex *dev_solution, 
			POINTER(c_complex),	# cuComplex *dev_new_z_extended, 
			POINTER(c_float),	# float *dev_gradient, 
			c_uint,			# const unsigned int h_index_of_max,
			c_uint			# const unsigned int N)
						]
