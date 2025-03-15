#include <stdio.h>
#include "/usr/include/gsl/gsl_math.h"
#include "/usr/include/gsl/gsl_linalg.h"

#define N 5  // Rozmiar macierzy i wektorów

// Deklaracje funkcji
void set_matrix(gsl_matrix *A);
void set_vector(gsl_vector *b);
void gauss_jordan(gsl_matrix *A, gsl_vector *b);
void matrix_vector_mult(gsl_matrix *A, gsl_vector *x, gsl_vector *c);
double deviation(gsl_vector *b, gsl_vector *c);
void set_matrix_A_ext(gsl_matrix *A_ext, gsl_matrix *A, gsl_vector *b);

int main(void)
{
    FILE *file = fopen("output.txt", "w");
    if (!file)
    {
        perror("Error opening file");
        return 1;
    }

    // Alokacja pamięci dla macierzy i wektorów
    gsl_matrix *A = gsl_matrix_calloc(N, N);
    gsl_vector *b = gsl_vector_calloc(N);
    gsl_vector *x = gsl_vector_calloc(N);
    gsl_vector *c = gsl_vector_calloc(N);

    set_matrix(A); // Inicjalizacja macierzy A
    set_vector(b); // Inicjalizacja wektora b

    // Iteracja po wartościach q
    for (double q = 0.01; q < 3; q += 0.01)
    {
        if (fabs(q - 1.0) < 1e-6) 
            continue;  // Pominięcie q = 1

        // Tworzenie rozszerzonej macierzy A_ext
        gsl_matrix *A_ext = gsl_matrix_calloc(N, N+1);
        gsl_matrix_set(A, 0, 0, 2 * q);
        set_matrix_A_ext(A_ext, A, b);

        // Rozwiązanie układu równań
        gauss_jordan(A_ext, x);

        // Obliczenie iloczynu macierzy A i wektora x
        matrix_vector_mult(A, x, c);

        // Obliczenie odchylenia od wektora b
        double o_q = deviation(b, c);

        // Zapis wyników do pliku
        fprintf(file, "vector b: \n");
        for (int i = 0; i < N; i++)
        {
            fprintf(file, "%.2f ", gsl_vector_get(b, i));
        }
        fprintf(file, "\nvector c: \n");
        for (int i = 0; i < N; i++)
        {
            fprintf(file, "%.2f ", gsl_vector_get(c, i));
        }
        fprintf(file, "\nq = %.2f o_q = %.15e\n", q, o_q);

        gsl_matrix_free(A_ext); // Zwolnienie pamięci
    }

    // Zamknięcie pliku i zwolnienie pamięci
    fclose(file);
    gsl_matrix_free(A);
    gsl_vector_free(b);
    gsl_vector_free(c);
    gsl_vector_free(x);

    return 0;
}

// Tworzy rozszerzoną macierz [A|b] poprzez dołączenie b jako ostatniej kolumny
void set_matrix_A_ext(gsl_matrix *A_ext, gsl_matrix *A, gsl_vector *b)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            gsl_matrix_set(A_ext, i, j, gsl_matrix_get(A, i, j));
        }
        gsl_matrix_set(A_ext, i, N, gsl_vector_get(b, i));
    }
}

// Implementacja metody Gaussa-Jordana do rozwiązania układu równań liniowych
void gauss_jordan(gsl_matrix *A, gsl_vector *x)
{
    for (int i = 0; i < N; i++)
    {
        double pivot = gsl_matrix_get(A, i, i);
        for (int j = 0; j < N + 1; j++)
        {
            double value = gsl_matrix_get(A, i, j) / pivot;
            gsl_matrix_set(A, i, j, value);
        }

        for (int k = 0; k < N; k++)
        {
            if (k != i)
            {
                double factor = gsl_matrix_get(A, k, i);
                for (int j = 0; j < N +1; j++)
                {
                    double value = gsl_matrix_get(A, k, j) - (factor * gsl_matrix_get(A, i, j));
                    gsl_matrix_set(A, k, j, value);
                }
            }
        }
    }

    // Wypełnienie wektora x wynikami
    for (int i = 0; i < N; i++)
    {
        gsl_vector_set(x, i, gsl_matrix_get(A, i, N));
    }
}

// Oblicza iloczyn macierzy A i wektora x, wynik zapisuje do c
void matrix_vector_mult(gsl_matrix *A, gsl_vector *x, gsl_vector *c)
{
    double value;
    for (int i = 0; i < N; i++)
    {
        gsl_vector_set(c, i, 0);
        value = 0.0;
        for (int j = 0; j < N; j++)
        {
            value += gsl_matrix_get(A, i, j) * gsl_vector_get(x, j);
            gsl_vector_set(c, i, value);
        }
    }
}

// Oblicza średnie odchylenie wektora c od wektora b
double deviation(gsl_vector *b, gsl_vector *c)
{
    double sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += ((gsl_vector_get(c, i) - gsl_vector_get(b, i)) *(gsl_vector_get(c, i) - gsl_vector_get(b, i)));
    }
    return sqrt(sum) * (1. / N);
}

// Inicjalizuje macierz A 
void set_matrix(gsl_matrix *A)
{
    double data[N][N] = {
        {0, 1, 6, 9, 10},
        {2, 1, 6, 9, 10},
        {1, 6, 6, 8, 6},
        {5, 9, 10, 7, 10},
        {3, 4, 9, 7, 9}
    };

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            gsl_matrix_set(A, i, j, data[i][j]);
        }
    }
}

// Inicjalizuje wektor b 
void set_vector(gsl_vector *b)
{
    double data[N] = {10, 2, 9, 9, 3};
    for (int i = 0; i < N; i++)
    {
        gsl_vector_set(b, i, data[i]);
    }
}
