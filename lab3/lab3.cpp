#include "iostream"
#include "fstream"
#include "math.h"
#include "/usr/include/gsl/gsl_math.h"
#include "/usr/include/gsl/gsl_linalg.h"

using namespace std;

const double TL = 1000;  // Temperatura na lewym końcu
const double TR = 100;   // Temperatura na prawym końcu
const double EPS = 1e-6; // Tolerancja błędu dla algorytmu
const int MAX_ITER = 60000; // Maksymalna liczba iteracji w algorytmie

void matrix_vector_mult(const gsl_matrix *A, const gsl_vector *x, gsl_vector *c);
double lambda_f(double x);
void build_matrix(gsl_matrix *A, gsl_vector *b, int n);
void print_matrix(gsl_matrix *A, int n);
void steepest_descent(const gsl_matrix *A, const gsl_vector *b, gsl_vector *t, ofstream &outFile);
void daxpy(double alpha, const gsl_vector *x, gsl_vector *y);
double vector_product(const gsl_vector *x, const gsl_vector *y);

int main()
{
    std::ofstream outFile("dane.txt"); 

    int n = 9; // Rozmiar macierzy 
    gsl_vector *b = gsl_vector_calloc(n); // Wektor b (po prawej stronie równania Ax = b)
    gsl_matrix *A = gsl_matrix_calloc(n, n); // Macierz A
    gsl_vector *t = gsl_vector_calloc(n); // Wektor rozwiązań t

    build_matrix(A, b, n); // Budowanie macierzy A i wektora b
    print_matrix(A, n); // Wypisywanie macierzy A
    steepest_descent(A, b, t, outFile); // Uruchomienie metody największych spadków

    cout << "Rozkład temperatur: \n";
    for (int i = 0; i < n; i++) {
        cout << gsl_vector_get(t, i) << " "; 
        outFile << (i + 1) / 10.  << " " << gsl_vector_get(t, i); // Zapisanie wyniku do pliku
        outFile << "\n"; 
    }
    cout << endl;

    gsl_matrix_free(A);
    gsl_vector_free(b);
    gsl_vector_free(t);

    return 0;
}

// Funkcja realizująca metodę największych spadków
void steepest_descent(const gsl_matrix *A, const gsl_vector *b, gsl_vector *t, ofstream &outFile)
{
    std::ofstream outFile1("wektor_reszt.txt"); 
    std::ofstream outFile2("wektor_temp.txt"); 
    int n = b->size;
    gsl_vector *r = gsl_vector_calloc(n); // Wektor reszty
    gsl_vector *Ar = gsl_vector_calloc(n); // Wektor Ax
    double alpha, rr, rAr; // Zmienna alpha (współczynnik krokowy), rr (iloczyn wektora r), rAr (iloczyn r i Ar)

    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        matrix_vector_mult(A, t, Ar); // Obliczenie Ax
        gsl_vector_memcpy(r, b); // r = b
        daxpy(-1.0, Ar, r); // r = b - Ax

        rr = vector_product(r, r); // Iloczyn skalarny r z r
        matrix_vector_mult(A, r, Ar); // Obliczenie A * r
        rAr = vector_product(r, Ar); // Iloczyn skalarny r z Ar

        alpha = rr / rAr; // Obliczenie współczynnika krokowego alpha
        daxpy(alpha, r, t); // t = t + alpha * r
        outFile1 << iter << " " << 0.5 * log10(rr) << std::endl;
        outFile2 << iter << " " << sqrt(vector_product(t, t)) << std::endl;

        // Warunek zakończenia iteracji
        if (sqrt(rr) < EPS)
            break;
    }

    gsl_vector_free(r);
    gsl_vector_free(Ar);
}

// Implementacja funkcji daxpy: y = alpha * x + y
void daxpy(double alpha, const gsl_vector *x, gsl_vector *y) 
{
    int n = x->size;
    for (int i = 0; i < n; i++) 
    {
        double value = alpha * gsl_vector_get(x, i) + gsl_vector_get(y, i);
        gsl_vector_set(y, i, value);
    }
}

// Implementacja iloczynu skalarnego: x * y
double vector_product(const gsl_vector *x, const gsl_vector *y)
{
    double result = 0.0;
    for (int i = 0; i < x->size; i++)
    {
        result += (gsl_vector_get(x, i) * gsl_vector_get(y, i));
    }

    return result;
}

// Mnożenie macierzy A przez wektor x: c = A * x
void matrix_vector_mult(const gsl_matrix *A, const gsl_vector *x, gsl_vector *c)
{
    double value;
    int n = x->size;
    for (int i = 0; i < n; i++)
    {
        value = 0.0;
        for (int j = 0; j < n; j++)
        {
            value += gsl_matrix_get(A, i, j) * gsl_vector_get(x, j);
        }
        gsl_vector_set(c, i, value);
    }
}

// Wypisywanie macierzy A na ekran
void print_matrix(gsl_matrix *A, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << gsl_matrix_get(A, i, j) << " ";
        }
        cout << endl;
    }
}

// Funkcja lambda_f określająca wartości współczynnika lambda w zależności od x
double lambda_f(double x)
{
    if (x <= 0.4)
    {
        return 0.3;
    }
    else if (x <= 0.7)
    {
        return 0.2;
    }
    else
        return 0.1;
}

// Funkcja budująca macierz A i wektor b na podstawie funkcji lambda_f
void build_matrix(gsl_matrix *A, gsl_vector *b, int n)
{
    double h = 1.0 / (n + 1);
    for (int i = 0; i < n; i++)
    {
        double xi = (i + 1) * h;
        double lam_1 = lambda_f(xi - 0.5 * h);
        double lam_2 = lambda_f(xi + 0.5 * h);

        gsl_matrix_set(A, i, i, -lam_1 - lam_2);
        if (i < n - 1)
            gsl_matrix_set(A, i, i + 1, lam_2);
        if (i > 0)
            gsl_matrix_set(A, i, i - 1, lam_1);
    }

    gsl_vector_set(b, 0, -lambda_f(0.5 * h) * TL); // Ustawienie wartości na lewym końcu
    gsl_vector_set(b, n - 1, -lambda_f(1 - 0.5 * h) * TR); // Ustawienie wartości na prawym końcu
}
