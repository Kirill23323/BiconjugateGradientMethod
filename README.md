# BiConjugate Gradient Solver (BiCG) for CRS Matrices (OpenMP)

BiConjugate Gradient (BiCG) is an iterative method for solving systems of linear algebraic equations with large sparse non-symmetric matrices. It is widely used in scientific computing, numerical simulations, and engineering applications.

This implementation is based on:
- CRS (CSR) sparse matrix format
- CSC format for efficient transpose multiplication (Aᵀx)
- OpenMP parallelization

---

## Features

- BiCG iterative solver for sparse linear systems
- CRS (CSR) matrix storage
- Automatic CSR → CSC conversion
- Efficient multiplication for A and Aᵀ
- OpenMP parallelization (dot product, SpMV, vector updates)
- Relative residual stopping criterion
- Scalable to large sparse matrices

---

## Requirements

Install dependencies:

```bash
sudo apt update
sudo apt install -y build-essential g++ libomp-dev
```

## Compilation & Run

### Compilation

To build the project with OpenMP support and оптимизацией:

```bash id="cmp1"
g++ -O3 -fopenmp -march=native bicg.cpp -o bicg
```


