echo SH
FOLDER=$1

echo nvcc -lcublas -lcufft -lcusolver -O3 --compiler-options '-fPIC' -shared ${FOLDER}/CUDA\ C\ ++\ sources/GMRES.cu -o ${FOLDER}/Shared\ object\ generating/TestLib.so
nvcc -lcublas -lcufft -lcusolver -O3 --compiler-options '-fPIC' -shared ${FOLDER}/CUDA\ C\ ++\ sources/GMRES.cu -o ${FOLDER}/Shared\ object\ generating/TestLib.so

