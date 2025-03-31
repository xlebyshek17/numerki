#include "iostream"
#include "fstream"
#include "math.h"
#include "vector"
#include "/usr/include/gsl/gsl_math.h"
#include "/usr/include/gsl/gsl_linalg.h"

#define N 7
#define M 12

using namespace std;

gsl_matrix* set_matrix();
gsl_vector* matrix_vector_mult(const gsl_matrix *A, const gsl_vector *x);
double vector_product(const gsl_vector* x, const gsl_vector* y);
void normalize(gsl_vector* x);
void reduce_matrix(gsl_matrix* A, const gsl_vector* x, double lambda);
void outer_product(const gsl_vector* x, gsl_matrix* result);
gsl_matrix* matrix_mult(const gsl_matrix *a, const gsl_matrix *b);
gsl_matrix* transpose(const gsl_matrix *x);
gsl_matrix* compute_matrix_D(const std::vector<gsl_vector*>& eigenvectors, gsl_matrix* A);

int main(void) 
{
    gsl_matrix *A = set_matrix();
    gsl_matrix *W = gsl_matrix_calloc(N, N);
    gsl_matrix *D = gsl_matrix_calloc(N, N);
    gsl_vector *x = gsl_vector_calloc(N);
    gsl_matrix_memcpy(W, A);

    vector<gsl_vector*> eigen_vectors;
    vector<double> eigen_values;

    for (int k = 0; k < N; k++)
    {
        gsl_vector_set_all(x, 1.0);

        //gsl_vector *new_x = gsl_vector_calloc(N);
        double lambda = 0.0;

        for (int iter = 0; iter < M; iter++)
        {
            gsl_vector *Ax = matrix_vector_mult(W, x);
            lambda = vector_product(Ax, x) / vector_product(x, x);
            normalize(Ax);
            gsl_vector_memcpy(x, Ax);
            gsl_vector_free(Ax);
        }

        eigen_vectors.push_back(gsl_vector_alloc(N));
        gsl_vector_memcpy(eigen_vectors.back(), x);
        eigen_values.push_back(lambda);

        reduce_matrix(W, x, lambda);
    }

    D = compute_matrix_D(eigen_vectors, A);

    std::ofstream out("matrix_D.txt");
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            out << gsl_matrix_get(D, i, j) << " ";
        }
        out << "\n";
    }
    out.close();

    std::ofstream output_matrix("eigen_matrix.txt");
    std::ofstream output_values("eigen_values.txt");
    output_matrix << "Macierz wektorów własnych:\n";
    output_values << "Wartości własne:\n";
    for (int i = 0; i < N; i++) 
    {
        output_values << eigen_values[i] << "\n";
        for (int j = 0; j < N; j++) 
        {
            output_matrix << gsl_vector_get(eigen_vectors[i], j) << " "; 
        }
        output_matrix << "\n";
    }
    output_matrix.close();
    output_values.close();

    gsl_matrix_free(A);
    gsl_matrix_free(W);
    gsl_matrix_free(D);

    for (auto vec : eigen_vectors) 
        gsl_vector_free(vec);

    std::cout << "Zakonczono diagonalizacje. Wyniki zapisano do pliku.\n";

    return 0;
}

gsl_matrix* compute_matrix_D(const std::vector<gsl_vector*>& eigenvectors, gsl_matrix* A)
{
    gsl_matrix *X = gsl_matrix_calloc(N, N);
    gsl_matrix *XT = gsl_matrix_calloc(N, N);
    gsl_matrix *temp = gsl_matrix_calloc(N, N);
    gsl_matrix *D = gsl_matrix_calloc(N, N);

    // Zbuduj X z wektorów własnych jako kolumn
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            gsl_matrix_set(X, j, i, gsl_vector_get(eigenvectors[i], j));
        }
    }

    XT = transpose(X);
    temp = matrix_mult(XT, A);
    D = matrix_mult(temp, X);
    return D;
}

gsl_matrix* transpose(const gsl_matrix *x)
{
    gsl_matrix *xt = gsl_matrix_calloc(N, N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) 
        {
            gsl_matrix_set(xt, i, j, gsl_matrix_get(x, j, i));
        }
    }

    return xt;
}

gsl_matrix* matrix_mult(const gsl_matrix *a, const gsl_matrix *b)
{
    gsl_matrix *res = gsl_matrix_calloc(N, N);
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += gsl_matrix_get(a, i, k) * gsl_matrix_get(b, k, j);
            }
            gsl_matrix_set(res, i, j, sum);
        }
    }

    return res;
}

// Iloczyn tensorowy x * x^T -> macierz
void outer_product(const gsl_vector* x, gsl_matrix* result) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            double value = gsl_vector_get(x, i) * gsl_vector_get(x, j);
            gsl_matrix_set(result, i, j, value);
        }
    }
}

// Redukcja macierzy: W - lambda * x * x^T
void reduce_matrix(gsl_matrix* A, const gsl_vector* x, double lambda) 
{
    gsl_matrix* xxT = gsl_matrix_alloc(N, N);
    outer_product(x, xxT);
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            double upd_value = gsl_matrix_get(A, i, j) - lambda * gsl_matrix_get(xxT, i, j);
            gsl_matrix_set(A, i, j, upd_value);
        }
    }
    gsl_matrix_free(xxT);
}

// Normalizacja wektora
void normalize(gsl_vector* x) 
{
    // Norma euklidesowa
    double n = sqrt(vector_product(x, x));

    for (int i = 0; i < N; i++)
    {
        gsl_vector_set(x, i, gsl_vector_get(x, i) / n);
    }
}

double vector_product(const gsl_vector* x, const gsl_vector* y) 
{
    double result = 0.0;
    for (int i = 0; i < N; i++) {
        result += gsl_vector_get(x, i) * gsl_vector_get(y, i);
    }
    return result;
}

// Mnożenie macierzy A przez wektor x
gsl_vector* matrix_vector_mult(const gsl_matrix *A, const gsl_vector *x)
{
    double value;
    int n = x->size;
    gsl_vector *result = gsl_vector_calloc(n);

    for (int i = 0; i < n; i++)
    {
        value = 0.0;
        for (int j = 0; j < n; j++)
        {
            value += gsl_matrix_get(A, i, j) * gsl_vector_get(x, j);
        }
        gsl_vector_set(result, i, value);
    }

    return result;
}

gsl_matrix* set_matrix()
{
    gsl_matrix *A = gsl_matrix_calloc(N, N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            gsl_matrix_set(A, i, j, pow((2 + abs(i - j)), (-abs(i - j) / 2.)));
        }
    }

    return A;
}