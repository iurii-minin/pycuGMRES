nvcc -lcublas -lcufft -lcusolver -O3 --compiler-options '-fPIC' -shared TestGMRES.cu -o ../Shared\ object\ generating/cuGMRES.so
