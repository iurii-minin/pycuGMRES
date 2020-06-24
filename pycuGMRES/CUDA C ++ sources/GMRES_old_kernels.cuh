__global__ void set_alpha_beta(cuComplex *cu_alpha, cuComplex *cu_beta)
{
	switch(blockIdx.x) 
	{
		case 0 :
		{
			cu_alpha->x = 1.f;
			break;
		}
		case 1 :
		{
			cu_alpha->y = 0.f;
			break;
		}
		case 2 :
		{
			cu_beta->x = 0.f;
			break;
		}
		case 3 :
		{
			cu_beta->y = 0.f;
			break;
		}
	}
}

__global__ void residual_vs_tolerance_kernel(float *residual, bool *res_vs_tol, const float tolerance)
{
	(*res_vs_tol) = ((*residual) > tolerance);
}


__global__ void next_residual_kernel(cuComplex *Jtotal_ij, float *norm_residual, float *actual_residual)
{
	*actual_residual =(*norm_residual) * sqrt( (pow((float)(Jtotal_ij->x), 2.0f) + pow((float)(Jtotal_ij->y), 2.0f)));
}




__global__ void set_inverse_kernel(const cuComplex *dev_H_, cuComplex *dev_temporary)
{
	switch(threadIdx.x)
	{
		case 0:
		{
			dev_temporary->x = - dev_H_->x;
			break;
		}		
		case 1:
		{
			dev_temporary->y = - dev_H_->y;
		}
	}
}

__global__ void create_Givens_rotation_matrix_kernel(cuComplex *dev_Givens_rotation, cuComplex *Htemp)
{
	unsigned int i = blockIdx.x;
	unsigned int j = blockIdx.y;
	unsigned int index = i * gridDim.y + j;

	if ((i < gridDim.x - 2) && (i == j))
	{
		dev_Givens_rotation[index].x = 1.f;
		dev_Givens_rotation[index].y = 0.f;
	}
	else
	{
		if ((i == gridDim.x - 2) && (j == gridDim.x - 2))
		{	
			unsigned int ind1 = index - i;
			unsigned int ind2 = index + 1;

			//fprintf(stderr, "1:\t%i\t%i\t%i\n", index, ind1, ind2);
			float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
			dev_Givens_rotation[index].x = Htemp[ind1].x / denominator;
			dev_Givens_rotation[index].y = Htemp[ind1].y / denominator;
		}
		else
		{	
			if ((i == gridDim.x - 2) && (j == gridDim.x - 1))
			{
				unsigned int ind2 = index - j;

				//fprintf(stderr, "2:\t%i\t%i\t%i\n", index, index, ind2);
				float denominator = sqrt(pow((float)Htemp[index].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[index].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
				dev_Givens_rotation[index].x = Htemp[index].x / denominator;
				dev_Givens_rotation[index].y = Htemp[index].y / denominator;
			}
			else
			{
				if ((i == gridDim.x - 1) && (j == gridDim.x - 2))
				{
					unsigned int ind1 = index - i;
					unsigned int ind2 = ind1  - i;


					//fprintf(stderr, "3:\t%i\t%i\t%i\n", index, ind1, ind2);
					float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
					dev_Givens_rotation[index].x = - Htemp[ind1].x / denominator;
					dev_Givens_rotation[index].y =   Htemp[ind1].y / denominator;
				}
				else
				{
					if ((i == gridDim.x - 1) && (j == gridDim.x - 1))
					{

						unsigned int ind2 = index - i - 1;
						unsigned int ind1 = ind2  - i;


						//fprintf(stderr, "4:\t%i\t%i\t%i\n", index, ind1, ind2);
						float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
						dev_Givens_rotation[index].x = Htemp[ind1].x / denominator;
						dev_Givens_rotation[index].y = Htemp[ind1].y / denominator;	
					}
					else
					{
						dev_Givens_rotation[index].x = 0.f;
						dev_Givens_rotation[index].y = 0.f;
					}
				}
			}
		}
	}
}

__global__ void Jtotal_resize_kernel(cuComplex *data, unsigned int current_size_ij, cuComplex *resized_data)
{
	unsigned int i = blockIdx.x;
	unsigned int j = blockIdx.y;
	unsigned int index_new = gridDim.y * i + j;

	if ((i < current_size_ij) && (j < current_size_ij))
	{
		int index_cur = current_size_ij * i + j;
		resized_data[index_new] = data[index_cur];
	}
	else
	{
		if ((i == gridDim.x - 1) && (i == j))
		{
			resized_data[index_new].x = 1.f;
			resized_data[index_new].y = 0.f;
		}
		else
		{
			resized_data[index_new].x = 0.f;
			resized_data[index_new].y = 0.f;
		}
	}
}


__global__ void resize_kernel(cuComplex *data, unsigned int current_size_i, unsigned int current_size_j, unsigned int new_size_j, cuComplex *resized_data)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	cuComplex zero_complex;
	zero_complex.x = 0.f;
	zero_complex.y = 0.f;
	resized_data[new_size_j * i + j] = ((i < current_size_i) && (j < current_size_j)) ? data[current_size_j * i + j] : zero_complex;
}

void get_resized(cuComplex **to_be_resized, dim3 gridsize, dim3 blocksize, unsigned int old_size_i, unsigned int old_size_j, unsigned int new_size_i, unsigned int new_size_j)
{
	cuComplex *dev_resized;	

	cudacall(cudaMalloc((void**)&dev_resized, new_size_i * new_size_j * sizeof(cuComplex)));
	
	resize_kernel <<< gridsize, blocksize >>> ((cuComplex *)(*to_be_resized), (unsigned int)old_size_i, (unsigned int)old_size_j, (unsigned int)new_size_j, (cuComplex *)dev_resized);
	cudacheckSYN();

	cudacall(cudaFree((cuComplex *)(*to_be_resized)));

	(*to_be_resized) = dev_resized;
}


void usual_MatMul_CUDA_Device_Mode(cublasHandle_t *handle, cuComplex *A, cuComplex *B, cuComplex *C, unsigned int n, unsigned int k, unsigned int m, const cuComplex *cu_alpha, const cuComplex *cu_beta)
{
	unsigned int lda = k, ldb = m;
	unsigned int ldc = (lda > ldb) ? ldb : lda;

	cublascall(cublasCgemm3m(*handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, cu_alpha, B, ldb, A, lda, cu_beta, C, ldc));
}


/*





__global__ void extend_by_zeros_kernel(bool *mask, cuComplex *usual, cuComplex *extended, const unsigned int N)
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
		if ((i < N) && (j < N) && (mask[index]))
		{
			current.x = CHI * usual[index].x;
			current.y = CHI * usual[index].y;
		}
		else
		{
			current.x = current.y = 0.f;
		}
		extended[index_extended] = current;
	}
}



__global__ void extend_by_zeros_kernel(cuComplex *usual, cuComplex *extended, const unsigned int N)
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
			current.x = CHI * usual[index].x;
			current.y = CHI * usual[index].y;
		}
		else
		{
			current.x = current.y = 0.f;
		}
		extended[index_extended] = current;
	}
}


__global__ void MatMul_ElemWise_Kernel(cuComplex *bttb_sur, cuComplex *vec2D, const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;

	if (( i < size_limit ) && ( j < size_limit ))
	{
		unsigned int index = size_limit * i + j;
		cuComplex curr_bttb = bttb_sur[index];
		cuComplex curr_out_mul = vec2D[index];
		vec2D[index].x = (curr_bttb.x * curr_out_mul.x - curr_out_mul.y * curr_bttb.y); // ((2 * N - 1) * (2 * N - 1));
		vec2D[index].y = (curr_out_mul.x * curr_bttb.y + curr_out_mul.y * curr_bttb.x); // ((2 * N - 1) * (2 * N - 1));
	}
}




__global__ void Fourier_normalize_kernel(cuComplex *dev_matmul_out_extended, const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;

	if (( i < size_limit ) && ( j < size_limit ))
	{
		unsigned int index = size_limit * i + j;
		cuComplex current = dev_matmul_out_extended[index];
		current.x /= ((2 * N - 1) * (2 * N - 1));
		current.y /= ((2 * N - 1) * (2 * N - 1));
		dev_matmul_out_extended[index] = current;
	}
}


__device__ __forceinline__ cuComplex my_cexpf(cuComplex z) //FOR ONLY z.x = 0.f;
{
	cuComplex res;
	sincosf(z.y, &res.y, &res.x);
	res.x *= E0;
	res.y *= E0;
	return res;
}

__global__ void _2D_to_1D_compared_kernel(bool *dev_mask, cuComplex *input_mul, cuComplex *_2D_in, cuComplex *residual, const unsigned int h_index_of_max, const unsigned int N)
{ 
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;


	if ((i < size_limit) && (j < size_limit))
	{
		unsigned int Ni = N * i;
		unsigned int _1D_index = Ni + j;
		unsigned int _2D_index = _1D_index + Ni - i;

		cuComplex current_2D = _2D_in[_2D_index];
		cuComplex arg_old = input_mul[_1D_index];

		if (dev_mask[_1D_index])
		{
			current_2D.x += (h_index_of_max == _1D_index) ? 1.f - arg_old.x : - arg_old.x;
			current_2D.y += - arg_old.y;
		}
		else
		{
			current_2D.x =  (h_index_of_max == _1D_index) ? 1.f - arg_old.x : - arg_old.x;
			current_2D.y = - arg_old.y;
		}

		residual[_1D_index] = current_2D;
	}
}


__global__ void _2D_to_1D_compared_kernel(cuComplex *input_mul, cuComplex *_2D_in, cuComplex *residual, const unsigned int N)
{ 
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;


	if ((i < size_limit) && (j < size_limit))
	{
		unsigned int Ni = N * i;
		unsigned int _1D_index = Ni + j;
		unsigned int _2D_index = _1D_index + Ni - i;
		cuComplex current_2D = _2D_in[_2D_index];
		cuComplex arg_old = input_mul[_1D_index];
		cuComplex Input_Field;

		Input_Field.y = - WAVE_NUMBER * (i * cos(ALPHA) + j * sin(ALPHA));
		Input_Field = my_cexpf(Input_Field);
		//float sigma = 400.f;
		//Input_Field.x = Input_Field.x * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
		//Input_Field.y = Input_Field.y * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
		//Input_Field.x = Input_Field.x * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
		//Input_Field.y = Input_Field.y * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
		current_2D.x += Input_Field.x - arg_old.x;
		current_2D.y += Input_Field.y - arg_old.y;
		residual[_1D_index] = current_2D;
	}
}


__global__ void _2D_to_1D_kernel(bool *dev_mask, cuComplex *input_mul, cuComplex *_2D_in, cuComplex *_1D_out, const unsigned int N)
{ 
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int Ni = N * i;
	unsigned int _1D_index = Ni + j;
	unsigned int _2D_index = _1D_index + Ni - i;
	cuComplex current = input_mul[_1D_index];
	cuComplex new_arg = _2D_in[_2D_index];

	if (dev_mask[_1D_index])
	{
		current.x -= new_arg.x;
		current.y -= new_arg.y;
	}
	_1D_out[_1D_index] = current;
}

__global__ void _2D_to_1D_kernel(cuComplex *input_mul, cuComplex *_2D_in, cuComplex *_1D_out, const unsigned int N)
{ 
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int Ni = N * i;
	unsigned int _1D_index = Ni + j;
	unsigned int _2D_index = _1D_index + Ni - i;
	cuComplex current = input_mul[_1D_index];
	cuComplex new_arg = _2D_in[_2D_index];
	
	current.x -= new_arg.x;
	current.y -= new_arg.y;

	_1D_out[_1D_index] = current;
}

__global__ void residual_normalization_kernel(cuComplex *residual_vec, float *norm_res_vec, cuComplex *dev_orthogonal_basis)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	cuComplex current = residual_vec[index];
	current.x = current.x / (*norm_res_vec);
	current.y = current.y / (*norm_res_vec);
	dev_orthogonal_basis[index] = current;
}



__global__ void weight_subtract_kernel(cuComplex *weight, cuComplex *Hjk, cuComplex *vj)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex current = weight[i];
	cuComplex current_vj = vj[i];
	
	current.x -= (Hjk->x) * current_vj.x - (Hjk->y) * current_vj.y;
	current.y -= (Hjk->y) * current_vj.x + (Hjk->x) * current_vj.y;
	weight[i] = current;
}


__global__ void get_complex_divided(const float *dev_Hjk_norm_part, cuComplex *dev_Hj, float *dev_divided)
{
	switch(blockIdx.x) 
	{
		case 0 :
		{	
			*dev_divided = 1.f / *dev_Hjk_norm_part;
			break;
		}
		case 1 :
		{
			dev_Hj->y      = 0.f;
			break;
		}
		case 2 :
		{
			dev_Hj->x = *dev_Hjk_norm_part;
			break;
		}
	}
}

__global__ void set_first_Jtotal_kernel(cuComplex *dev_Jtotal, cuComplex *Htemp, const unsigned int characteristic_size)
{
	switch(blockIdx.x)
	{	
		case 0 :
		{
			float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[1].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[1].y, 2.f));
			dev_Jtotal[0].x = Htemp[0].x / denominator;
			break;
		}	
		case 1 :
		{
			float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[1].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[1].y, 2.f));
			dev_Jtotal[0].y = Htemp[0].y / denominator;
			break;
		}
		case 2 :
		{
			float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[1].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[1].y, 2.f));
			dev_Jtotal[1].x = Htemp[1].x / denominator;
			break;
		}
		case 3 :
		{
			float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[1].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[1].y, 2.f));
			dev_Jtotal[1].y = Htemp[1].y / denominator;
			break;
		}
		default:
		{
			switch(blockIdx.x - characteristic_size >> 1)
			{
				case 0:
				{
					float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[1].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[1].y, 2.f));
					dev_Jtotal[characteristic_size].x = - Htemp[1].x / denominator;
					break;
				}
				case 1:
				{
					float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[1].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[1].y, 2.f));
					dev_Jtotal[characteristic_size].y =   Htemp[1].y / denominator;
					break;
				}
				case 2:
				{
					float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[1].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[1].y, 2.f));
					dev_Jtotal[characteristic_size + 1].x = Htemp[0].x / denominator;
					break;
				}
				case 3:
				{
					float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[1].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[1].y, 2.f));
					dev_Jtotal[characteristic_size + 1].y = Htemp[0].y / denominator;
					break;
				}
				default:
				{
					if (blockIdx.x % 2)
					{
						dev_Jtotal[blockIdx.x / 2].x = blockIdx.x / 2 % (characteristic_size + 1) ? 0.f : 1.f;
					}else
					{
						dev_Jtotal[blockIdx.x / 2].y = 0.f;
					}
				}
			}
		}		
	}
}



__global__ void set_Identity_matrix_kernel(cuComplex *dev_Identity_matrix)
{
	unsigned int i = blockIdx.x;
	unsigned int j = blockIdx.y;
	cuComplex current;

	current.x = i == j ? 1.f : 0.f;
	current.y = 0.f; 

	dev_Identity_matrix[i * gridDim.x + j] = current;
}









__global__ void set_4_Givens_rotation_matrix_elements_kernel(	cuComplex *dev_Givens_rotation,
								cuComplex *dev_Htemp,
								const unsigned int characteristic_size,
								const unsigned int index_J_0,
								const unsigned int index_J_1,
								const unsigned int index_J_2,
								const unsigned int index_J_3,
								const unsigned int GMRES_i_plus_1)
{

	const unsigned int index_H_0 = GMRES_i_plus_1 * GMRES_i_plus_1 - 1;
	const unsigned int index_H_1 = index_H_0 + GMRES_i_plus_1;

	float denominator = sqrt(pow((float)dev_Htemp[index_H_0].x, 2.f) + pow((float)dev_Htemp[index_H_1].x, 2.f) + pow((float)dev_Htemp[index_H_0].y, 2.f) + pow((float)dev_Htemp[index_H_1].y, 2.f));

	switch(blockIdx.x)
	{
		case 0:
		{
			dev_Givens_rotation[index_J_0].x = dev_Htemp[index_H_0].x / denominator;
			break;
		}
		case 1:
		{
			dev_Givens_rotation[index_J_0].y = dev_Htemp[index_H_0].y / denominator;
			break;
		}
		case 2:
		{
			dev_Givens_rotation[index_J_1].x = dev_Htemp[index_H_1].x / denominator;
			break;
		}
		case 3:
		{
			dev_Givens_rotation[index_J_1].y = dev_Htemp[index_H_1].y / denominator;
			break;
		}
		case 4:
		{
			dev_Givens_rotation[index_J_2].x = - dev_Htemp[index_H_1].x / denominator;
			break;
		}
		case 5:
		{
			dev_Givens_rotation[index_J_2].y =   dev_Htemp[index_H_1].y / denominator;
			break;
		}
		case 6:
		{
			dev_Givens_rotation[index_J_3].x = dev_Htemp[index_H_0].x / denominator;
			break;
		}
		case 7:
		{
			dev_Givens_rotation[index_J_3].y = dev_Htemp[index_H_0].y / denominator;
		}
	}
}

__global__ void set_cc_kernel(cuComplex *cc, cuComplex *Jtotal, float *old_norm_res_vec, const unsigned int characteristic_size)
{	
	unsigned int index = blockIdx.x * characteristic_size;

	cc[blockIdx.x].x = Jtotal[index].x * (*old_norm_res_vec);
	cc[blockIdx.x].y = Jtotal[index].y * (*old_norm_res_vec);
}

__global__ void get_new_solution_kernel(cuComplex *dev_cc, cuComplex *dev_HH)
{
	float dominant = pow((float)(dev_HH->x), 2.f) + pow((float)(dev_HH->y), 2.f);
	cuComplex current;
	current.x = (dev_cc->x * dev_HH->x + dev_cc->y * dev_HH->y) / dominant;
	current.y = (dev_cc->y * dev_HH->x - dev_cc->x * dev_HH->y) / dominant;
	(*dev_cc) = current;
}

__global__ void get_solution_kernel(cuComplex *dev_solution, cuComplex *dev_cc, cuComplex *dev_orthogonal_basis)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex current = dev_orthogonal_basis[index];
	atomicAdd((float *)&(dev_solution[index].x), current.x * dev_cc->x - current.y * dev_cc->y);
	atomicAdd((float *)&(dev_solution[index].y), current.x * dev_cc->y + current.y * dev_cc->x);
}

__global__ void add_kernel(cuComplex *dev_solution, cuComplex *dev_add_x, cuComplex *mul)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd((float *)&(dev_solution[index].x), mul->x * dev_add_x[index].x - mul->y * dev_add_x[index].y);
	atomicAdd((float *)&(dev_solution[index].y), mul->y * dev_add_x[index].x + mul->x * dev_add_x[index].y);
}

__global__ void init_x0_kernel(cuComplex *input, const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	cuComplex Input_Field;

	Input_Field.y = - WAVE_NUMBER * (i * cos(ALPHA) + j * sin(ALPHA));
	Input_Field = my_cexpf(Input_Field);
	//float sigma = 400.f;
	//Input_Field.x = Input_Field.x * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
	//Input_Field.y = Input_Field.y * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
	//Input_Field.x = Input_Field.x * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
	//Input_Field.y = Input_Field.y * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
	input[i * N + j] = Input_Field;
}



__global__ void create_Givens_rotation_matrix_kernel(cuComplex *dev_Givens_rotation, cuComplex *Htemp)
{
	unsigned int i = blockIdx.x;
	unsigned int j = blockIdx.y;
	unsigned int index = i * gridDim.y + j;

	if ((i < gridDim.x - 2) && (i == j))
	{
		dev_Givens_rotation[index].x = 1.f;
		dev_Givens_rotation[index].y = 0.f;
	}
	else
	{
		if ((i == gridDim.x - 2) && (j == gridDim.x - 2))
		{	
			unsigned int ind1 = index - i;
			unsigned int ind2 = index + 1;

			//fprintf(stderr, "1:\t%i\t%i\t%i\n", index, ind1, ind2);
			float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
			dev_Givens_rotation[index].x = Htemp[ind1].x / denominator;
			dev_Givens_rotation[index].y = Htemp[ind1].y / denominator;
		}
		else
		{	
			if ((i == gridDim.x - 2) && (j == gridDim.x - 1))
			{
				unsigned int ind2 = index - j;

				//fprintf(stderr, "2:\t%i\t%i\t%i\n", index, index, ind2);
				float denominator = sqrt(pow((float)Htemp[index].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[index].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
				dev_Givens_rotation[index].x = Htemp[index].x / denominator;
				dev_Givens_rotation[index].y = Htemp[index].y / denominator;
			}
			else
			{
				if ((i == gridDim.x - 1) && (j == gridDim.x - 2))
				{
					unsigned int ind1 = index - i;
					unsigned int ind2 = ind1  - i;


					//fprintf(stderr, "3:\t%i\t%i\t%i\n", index, ind1, ind2);
					float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
					dev_Givens_rotation[index].x = - Htemp[ind1].x / denominator;
					dev_Givens_rotation[index].y =   Htemp[ind1].y / denominator;
				}
				else
				{
					if ((i == gridDim.x - 1) && (j == gridDim.x - 1))
					{

						unsigned int ind2 = index - i - 1;
						unsigned int ind1 = ind2  - i;


						//fprintf(stderr, "4:\t%i\t%i\t%i\n", index, ind1, ind2);
						float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
						dev_Givens_rotation[index].x = Htemp[ind1].x / denominator;
						dev_Givens_rotation[index].y = Htemp[ind1].y / denominator;	
					}
					else
					{
						dev_Givens_rotation[index].x = 0.f;
						dev_Givens_rotation[index].y = 0.f;
					}
				}
			}
		}
	}
}





*/
