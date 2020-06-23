nvcc -lcublas -lcufft -lcusolver -O3 --compiler-options '-fPIC' -shared TestGMRES.cu -o ../Shared\ object\ generating/cuGMRES.so

while true; do
    read -p "Do you wish to update this program?" yn
    case $yn in
        [Yy]* ) cd ../../..; bash all_update.sh; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
