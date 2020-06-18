FOLDER=$1
soFILE="${FOLDER}/Shared object generating/cuGMRES.so"

if test -f "$soFILE"; then
    rm $soFILE
fi

echo importing pycuGMRES ...
echo nvcc -lcublas -lcufft -lcusolver -O3 --compiler-options '-fPIC' -shared \"CUDA\ C\ ++\ sources\" -o cuGMRES.so
nvcc -lcublas -lcufft -lcusolver -O3 --compiler-options '-fPIC' -shared ${FOLDER}/CUDA\ C\ ++\ sources/TestGMRES.cu -o $soFILE
if test -f "$soFILE"; then
    echo CUDA C++ library has been created!
else 
    echo FATAL ERROR: CUDA C++ library creating failed!
fi

