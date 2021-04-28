FOLDER=$1
soFILE="${FOLDER}/Shared object generating/cuGMRES.so"
echo importing pycuGMRES 

if test -f "$soFILE"; then
    rm "$soFILE"
fi

echo nvcc -lcublas -lcufft -lcusolver -O3 --compiler-options '-fPIC' -shared \"CUDA\ C\ ++\ sources\" -o cuGMRES.so
nvcc -lcublas -lcufft -lcusolver -O3 --compiler-options '-fPIC' -shared ${FOLDER}/CUDA\ C\ ++\ sources/TestGMRES.cu -o "$soFILE"
if test -f "$soFILE"; then
    echo pycuGMRES library is ready to use!
else 
    echo FATAL ERROR: CUDA C++ - based library creating failed!
fi

