version = "1.1.4.5.8"

from setuptools import setup

with open("README.md", "r") as fh:
       long_description = fh.read()

setup(name='pycuGMRES',
      version = version,
      description = 'Fast CUDA C++ GMRES implementation for Toeplitz-like (Toeplitz, Hankel, Circulant) matrices and mixed (combinations of Diagonal ones and Toeplitz-like ones) matrices.',
      long_description = long_description,
      url='https://github.com/archilless/pycuGMRES',
      author='Iurii Borisovich Minin',
      author_email='iurii.minin@skoltech.ru', #yurii.minin@phystech.edu
      license='MIT',
      packages=['pycuGMRES'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      include_package_data = True,
      package_data = {'pycuGMRES': ['*.cu', '*.sh', '*.cuh', '*.so', 'CUDA C ++ sources/*.cu', 'CUDA C ++ sources/*.cuh', 'Shared object generating/*.sh']
}
)
