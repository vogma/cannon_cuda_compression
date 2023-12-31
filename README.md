# Cannon's Algorithm implemented with Cuda-Aware MPI and message compression

[![DOI](https://zenodo.org/badge/668173240.svg)](https://zenodo.org/badge/latestdoi/668173240)

Implementation of cannon's algorithm using MPI and CUDA to explore the use of Compression in Cuda-Aware MPI Applications. The compression relies on [ndzip](https://github.com/celerity/ndzip), which is included in this repository as a submodule


## Prerequisites
- CMake >= 3.18
- GCC >= 10.3.0
- Linux (tested on x86_64)
- Boost >= 1.66
- Cuda >= 11.3
- Cuda-Aware MPI Implementation (tested on OpenMPI v4.1.1)

## Building


```sh
mkdir build
cmake -B build .
cmake --build build -j 
```

two executables will be build:
- Cannons_Algorithm
- Cannons_Algorithm_Comp

Both can be used for distributed matrix multiplication using cannon's algorithm. Cannons_Algorithm_Comp compresses the subblocks with ndzip before sending them to another node.
The receiving node will then decompress the data. 

## Running

The application takes two parameters: 
- size of the matrizes
- path to file with testdata

The first parameter sets the size of the matrizes. For example with a given argument of 8192 the programm will execute a distributed multiplication of two matrizes each with a size of 8192 x 8192. 

The second parameter expects a path to a binary file with double precision floating point data. With this data the matrizes will be populated. The layout of this data file has to be the same as the double precision floating point datasets provided by Martin Burtscher [[Link](https://userweb.cs.txstate.edu/~burtscher/research/datasets/FPdouble/)].

This implementation has been tested on the HPC System [JUSUF](https://www.fz-juelich.de/en/ias/jsc/systems/supercomputers/jusuf), located at the Research Center in Jülich. 

An example SLURM Jobscript is provided which was used for testing on JUSUF.
