#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <math.h>

#define N 4
#define DELTA 2

// Funkcja do rozkładu LU
void lu_decompose(gsl_matrix *A, gsl_matrix *L, gsl_matrix *U) {
    int s;
    gsl_permutation *p = gsl_permutation_alloc(A->size1);
    
    gsl_linalg_LU_decomp(A, p, &s);  // Rozkład LU
    
    gsl_linalg_LU_get(L, U, A, p);
    
    gsl_permutation_free(p);
}

// Funkcja do obliczania odwrotności macierzy
void inverse_matrix(gsl_matrix *A, gsl_matrix *A_inv) {
    int s;
    gsl_permutation *p = gsl_permutation_alloc(A->size1);
    
    gsl_linalg_LU_decomp(A, p, &s);
    gsl_linalg_LU_invert(A, p, A_inv);
    
    gsl_permutation_free(p);
}

// Funkcja do mnożenia macierzy
void multiply_matrices(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C) {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}

// Funkcja do obliczania wskaźnika uwarunkowania
// double condition_number(gsl_matrix *A) {
//     double norm_A = gsl_linalg_norm(A, gsl_norm_1);
//     gsl_matrix *A_inv = gsl_matrix_alloc(A->size1, A->size2);
    
//     inverse_matrix(A, A_inv);
//     double norm_A_inv = gsl_linalg_norm(A_inv, gsl_norm_1);
    
//     gsl_matrix_free(A_inv);
    
//     return norm_A * norm_A_inv;
// }

// Funkcja do zapisywania wyników do pliku
void save_matrix_to_file(const char *filename, gsl_matrix *matrix) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Nie można otworzyć pliku");
        return;
    }
    for (size_t i = 0; i < matrix->size1; i++) {
        for (size_t j = 0; j < matrix->size2; j++) {
            fprintf(file, "%8.2f ", gsl_matrix_get(matrix, i, j));
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main() {
    // Przykładowa macierz A 4x4
    gsl_matrix *A = gsl_matrix_alloc(N, N);
    
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            long double value = 1.L / (i + j + DELTA);
            gsl_matrix_set(A, i, j, (double)value);
        }
    }

    save_matrix_to_file("matrix_A.txt", A);
    
    // Macierz L i U
    gsl_matrix *L = gsl_matrix_alloc(N, N);
    gsl_matrix *U = gsl_matrix_alloc(N, N);
    
    // Rozkład LU
    lu_decompose(A, L, U);

    save_matrix_to_file("decompose_L.txt", L);
    save_matrix_to_file("decompose_U.txt", U);
    
    // Obliczanie odwrotności
    gsl_matrix *A_inv = gsl_matrix_alloc(N, N);
    inverse_matrix(A, A_inv);

    save_matrix_to_file("inverxe_mtarix.txt", A_inv);
    
    // Mnożenie macierzy
    gsl_matrix *C = gsl_matrix_alloc(N, N);
    multiply_matrices(A, A_inv, C);

    save_matrix_to_file("matrix_C.txt", C);
    
    // // Obliczanie wskaźnika uwarunkowania
    // double cond = condition_number(A);
    // printf("Wskaźnik uwarunkowania: %g\n", cond);
    
    // // Zapisywanie wyników do pliku
    // save_to_file("output.txt", C);
    
    // // Zwalnianie pamięci
    // gsl_matrix_free(A);
    // gsl_matrix_free(L);
    // gsl_matrix_free(U);
    // gsl_matrix_free(A_inv);
    // gsl_matrix_free(C);
    
    return 0;
}