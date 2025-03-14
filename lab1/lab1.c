#include <stdio.h>
#include "/usr/include/gsl/gsl_math.h"
#include "/usr/include/gsl/gsl_linalg.h"

#define N 4 // rpzmiar macierzy
#define DELTA 2

// Funkcja mnożąca macierze A * B
void multiply_matrices(gsl_matrix *A, gsl_matrix *B, gsl_matrix *result) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < N; k++) {
                sum += gsl_matrix_get(A, i, k) * gsl_matrix_get(B, k, j);
            }
            gsl_matrix_set(result, i, j, sum);
        }
    }
}

void save_diagonal_and_determiniant(const gsl_matrix *a, FILE *file)
{
    double determinant  = 1;

    fprintf(file, "Elementy diagonalne macierzy: \n");
    for (int i = 0; i < a->size1; i++)
    {
        double diag = gsl_matrix_get(a, i, i);
        fprintf(file, "%8.3f\n", diag);
        determinant *= diag;
    }
    fprintf(file, "Wyznacznik macierzy = %8.3f", determinant);
}

// Funkcja zapisująca macierz do pliku
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

int main(void)
{
    gsl_matrix *A = gsl_matrix_alloc(N, N);
    gsl_matrix *C = gsl_matrix_calloc(N, N);
    gsl_vector *b = gsl_vector_alloc(N);
    gsl_vector *x = gsl_vector_alloc(N);
    gsl_permutation *p = gsl_permutation_alloc(N);

    int signum;

    // Definicja macierzy A
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            long double value = 1.L / (i + j + DELTA);
            gsl_matrix_set(A, i, j, (double)value);
            printf("%8.3f ", gsl_matrix_get(A, i, j));
        }
        printf("\n");
    }
    printf("\n");

    // Faktoryzacja LU
    gsl_linalg_LU_decomp(A, p, &signum);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%8.3f ", gsl_matrix_get(A, i, j));
        }
        printf("\n");
    }
    printf("\n");

    FILE *file = fopen("diagonal_and_determinant.txt", "w");
    if (file)
    {
        save_diagonal_and_determiniant(A, file);
        fclose(file);
        printf("Dane zapisane do pliku diagonal_and_determinant.txt\n");
    }
    else
        printf("blad zapisu");

    
    // Rozwiązanie układu Ax = b
    gsl_vector_set_all(b, 1.0); // Wektor wyrazów wolnych
    gsl_linalg_LU_solve(A, p, b, x);

    // Wypisanie wyniku
    printf("Rozwiązanie układu Ax = b:\n");
    for (int i = 0; i < N; i++) {
        printf("x[%d] = %f\n", i, gsl_vector_get(x, i));
    }

    gsl_matrix *A_inv = gsl_matrix_alloc(N, N);
    gsl_linalg_LU_invert(A, p, A_inv);
    
    // Zapis macierzy odwrotnej do pliku
    save_matrix_to_file("inverse_matrix.txt", A_inv);

    multiply_matrices(A, A_inv, C);

    save_matrix_to_file("iloczyn_skalarny.txt", C);

    gsl_matrix_free(A);
    gsl_permutation_free(p);
    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_matrix_free(A_inv);
    gsl_matrix_free(C);

    return 0;
}