nvcc -lcublas -lcufft -lcusolver -O3 --compiler-options '-fPIC' -shared GMRES.cu -o ../TestLib.so && cd .. && python3 cuGMRES.py
