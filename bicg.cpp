#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <cstring>
#include <omp.h>
#include <algorithm>
#include <vector>


struct CRSMatrix
{
     int n; 
     int m; 
     int nz; 
    std:: vector<double> val; 
    std:: vector<int> colIndex; 
    std:: vector<int> rowPtr; 
};


CRSMatrix CSR_to_CSC(const CRSMatrix& A)
{
    CRSMatrix AT;
    AT.n = A.m;   
    AT.m = A.n;
    AT.nz = A.nz;
    AT.val.resize(A.nz);
    AT.colIndex.resize(A.nz);
    AT.rowPtr.assign(A.m + 1, 0);
    for (int i = 0; i < A.nz; ++i)
    {
        int col = A.colIndex[i];
        AT.rowPtr[col + 1]++;
    }
    for (int i = 0; i < A.m; ++i)
    {
        AT.rowPtr[i + 1] += AT.rowPtr[i];
    }
    std::vector<int> pos = AT.rowPtr;
    for (int row = 0; row < A.n; ++row)
    {
        for (int i = A.rowPtr[row]; i < A.rowPtr[row + 1]; ++i)
        {
            int col = A.colIndex[i];
            double value = A.val[i];

            int dest = pos[col]++;

            AT.colIndex[dest] = row;   
            AT.val[dest] = value;
        }
    }

    return AT;
}


double Dot(const double* a,const double* b,int n) {

    double sum = 0;
#pragma omp parallel for reduction(+:sum) 
    for(int i = 0;i < n;++i){
        sum += a[i] * b[i];
    }
    return sum;

}


void Smdv(CRSMatrix& A, double* result, const double* v) { 
    
    std::fill(result, result + A.n, 0.0); 
#pragma omp parallel for
    for(int row = 0;row < A.n;++row) {
        double sum = 0.0;      
        for(int i = A.rowPtr[row];i < A.rowPtr[row + 1]; ++i) {
            sum += A.val[i] * v[A.colIndex[i]];
        }
        result[row] = sum;
    }

}


double Norm(const double* vector,int n) {

    double sum = 0;
#pragma omp parallel for reduction(+:sum) 
    for(int i = 0;i < n;++i) {
        sum += vector[i] * vector[i];
    }
    return std::sqrt(sum);

}


void SLE_Solver_CRS_BICG(CRSMatrix & A, double * b, double eps, int max_iter, double * x, int & count) {
    
    const double eps_div = 1e-15; 
    const double eps_relative = eps; 

    std::memset(x,0,static_cast<size_t>(A.n * sizeof(double))); 

    double* r = new double[A.n]();
    double* r_ = new double[A.n]();
    double* p = new double[A.n]();
    double* p_ = new double[A.n]();
    double* Ap = new double[A.n](); 
    double* Atp_ = new double[A.n](); 

    std::memcpy(r,b,static_cast<size_t>((A.n) * sizeof(double))); 

#pragma omp parallel for
    for(int row = 0;row < A.n;++row) {      
        for(int i = A.rowPtr[row];i < A.rowPtr[row + 1]; ++i) {
            r[row] -= A.val[i] *  x[A.colIndex[i]];
        }
    }

    double norm_b = Norm(b, A.n);
    if (norm_b < eps_div){ 
        norm_b = 1.0; 
    } 

    std::memcpy(r_,r,static_cast<size_t>((A.n) * sizeof(double))); 
    std::memcpy(p,r,static_cast<size_t>((A.n) * sizeof(double))); 
    std::memcpy(p_,r_,static_cast<size_t>((A.n) * sizeof(double)));

    CRSMatrix AT = CSR_to_CSC(A); 
  
    for(int j = 0;j < max_iter;++j) {
        double dot_r = Dot(r,r_,A.n);
        if (std::abs(dot_r) < eps_div) {
            break; 
        }
        Smdv(A,Ap,p);
        double dot_App_ = Dot(Ap,p_,A.n);
        if (std::abs(dot_App_) < eps_div) {
            break;
        }
        double alpha = dot_r / dot_App_; 
    #pragma omp parallel for
        for (int i = 0; i < A.n; ++i) {
            x[i] += alpha * p[i];
        }
        
    #pragma omp parallel for
        for (int i = 0; i < A.n; ++i) {
            r[i] -= alpha * Ap[i];
        }
        double norm_r = Norm(r, A.n);
         ++count;
        if (std::isnan(norm_r) || std::isinf(norm_r) || (norm_r / norm_b < eps_relative)) {
            break; 
        }
        Smdv(AT,Atp_,p_);
    #pragma omp parallel for
        for (int i = 0; i < A.n; ++i) {
            r_[i] -= alpha * Atp_[i];
        }
        double dot_new_r = Dot(r,r_,A.n);

        if (std::abs(dot_r) < eps_div) {
            break;
        }
        double beta = dot_new_r / dot_r;
    #pragma omp parallel for
        for (int i = 0; i < A.n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
    #pragma omp parallel for
        for (int i = 0; i < A.n; ++i) {
            p_[i] = r_[i] + beta * p_[i];
        }
    }
    
    delete [] Ap; 
    delete [] Atp_; 
    delete [] r;
    delete [] r_;
    delete [] p;
    delete [] p_;
}

void GenerateTestMatrix(CRSMatrix& A, int n, int elements_per_row) {
    A.n = n;
    A.m = n;

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> col_dist(0, n - 1);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0);

    A.rowPtr.resize(n + 1);
    A.rowPtr[0] = 0;

    for (int i = 0; i < n; ++i) {
        std::vector<int> used;

        for (int j = 0; j < elements_per_row; ++j) {
            int col = col_dist(gen);
            used.push_back(col);

            A.colIndex.push_back(col);
            A.val.push_back(val_dist(gen));
        }

        A.colIndex.push_back(i);
        A.val.push_back(elements_per_row + 5.0);

        A.rowPtr[i + 1] = A.colIndex.size();
    }

    A.nz = A.val.size();
}

void PerformanceTest(int n) {
    CRSMatrix A;
    GenerateTestMatrix(A, n, 10);

    std::vector<double> b(n, 1.0);
    std::vector<double> x(n, 0.0);

    int count = 0;

    auto start = std::chrono::high_resolution_clock::now();

    SLE_Solver_CRS_BICG(A, b.data(), 1e-6, 10000, x.data(), count);

    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(end - start).count();

    std::cout << "Размер: " << n << std::endl;
    std::cout << "Итерации: " << count << std::endl;
    std::cout << "Время: " << time << " сек" << std::endl;

    // ||Ax - b||
    std::vector<double> r(n, 0.0);
    Smdv(A, r.data(), x.data());

    for (int i = 0; i < n; ++i)
        r[i] -= b[i];

    std::cout << "||Ax - b|| = " << Norm(r.data(), n) << std::endl;
}

void Benchmark() {
    std::vector<int> sizes = {1000, 5000, 10000, 20000};

    for (int n : sizes) {
        std::cout << "====================\n";
        PerformanceTest(n);
    }
}

int main() {

    Benchmark();
    return 0;
}
