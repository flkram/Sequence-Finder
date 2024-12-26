# CUDA Sequence Finder

This program utilizes CUDA to find the number of sequences in a list of numbers that add up to a target number. The program leverages the parallel processing capabilities of the GPU by splitting each sequence search into multiple threads, significantly speeding up the computation for large datasets.

## Usage
```bash
./sequence [target num] [optional report flag]
```

#### Examples (using given input files)
```bash
./sequence 10 report < example-input-1.txt
```
```bash
./sequence 100 < example-input-3.txt
```

#### Compilation
```bash
nvcc -o sequence sequence.cu
```

## Features

- **GPU Parallelization**: CUDA is used to parallelize the search for sequences, making the program efficient for large datasets.
- **Flexible Input**: The input consists of a list of numbers from a file, and the program searches for sequences that add up to a given target number.
- **Optional Reporting**: A report flag can be passed to print detailed information about the sequences that contribute to the sum.

## Requirements

- **CUDA Toolkit**: Ensure that you have the CUDA Toolkit installed and your environment is configured to compile and run CUDA programs. Otherwise program will return "Failure in CUDA kernel execution".


