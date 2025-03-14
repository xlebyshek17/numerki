#include <iostream>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <fstream>
int n = 5; 
std::ofstream MyFile("lab02.txt");
// funkcja mnozenia macierzy 
void multiply(gsl_matrix *A, gsl_vector *x, gsl_vector *c) {
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int k = 0; k < n; k++) {
            sum += gsl_matrix_get(A, i, k) * gsl_vector_get(x, k);
        }
        gsl_vector_set(c, i, sum); 
    }
}
// funkcja sumy pomocniczej
double sum(gsl_vector *b,  gsl_vector *c){
    double sum=0;
    for(int i=0; i<n; i++){
        sum+=(gsl_vector_get(c, i)-gsl_vector_get(b, i))*(gsl_vector_get(c, i)-gsl_vector_get(b, i));
    }
    return sum;
}
// Funkcja obliczająca odchylenie
double counting(gsl_vector *b,  gsl_vector *c){
    return (1.0/5)*sqrt((sum(b,c)));
}
//układ równań liniowych metodą Gaussa-Jordana
void gaus_jordan(gsl_matrix *A, gsl_vector *b, gsl_vector *x, gsl_matrix *answ){
    for (int i=0; i<n; i++){
        for(int j=0; j<n ;j++){
            gsl_matrix_set(answ, i, j, gsl_matrix_get(A, i, j));
        }
    }    
    for(int i=0; i<n; i++){
         gsl_matrix_set(answ, i, n, gsl_vector_get(b, i));
    } 

    for (int i = 0; i < n; i++) {
        double a11 = gsl_matrix_get(answ, i, i);
        for (int j = 0; j < n + 1; j++) {
             gsl_matrix_set(answ, i, j, gsl_matrix_get(answ, i, j) / a11);
        }
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = gsl_matrix_get(answ, k, i);
                for (int j = 0; j < n + 1; j++) {
                    gsl_matrix_set(answ, k, j, gsl_matrix_get(answ, k, j) - factor * gsl_matrix_get(answ, i, j));
                }
            }
        }
    }
    for(int i=0; i<n; i++){
        gsl_vector_set(x, i, gsl_matrix_get(answ, i, n));
   } 
}

void print(gsl_matrix *A){
        for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << gsl_matrix_get(A, i, j) << " ";
        }
        std::cout << std::endl;
    }

}
void print_answ(gsl_matrix *answ){
    for (int i = 0; i < n; i++) {
    for (int j = 0; j < n+1; j++) {
        std::cout << gsl_matrix_get(answ, i, j) << " ";
        MyFile<<gsl_matrix_get(answ, i, j) << " ";
    }
    MyFile<< std::endl;
    std::cout << std::endl;
}

}



int main(){
   
    gsl_matrix *A = gsl_matrix_calloc(n, n);
    gsl_vector *b = gsl_vector_calloc(n);
    gsl_vector *x = gsl_vector_calloc(n);
    gsl_matrix *answ = gsl_matrix_calloc(n, n+1);
    gsl_vector *c = gsl_vector_calloc(n);
        double inside_b[n]={10,2,9,9, 3};
    for (int i=0; i<n; i++){
        gsl_vector_set(b, i, inside_b[i]);
    }
 
    //petla ktora tworzy rozne macierzy
    for(double q=0.01; q<2.99; q+=0.01){
        if (fabs(q - 1.0) < 1e-6) continue;
        double A_inside[n][n]={
                {2*q, 1, 6, 9, 10},
                {2,1,6,9,10},
                {1,6,6,8,6},
                {5,9,10,7,10},
                {3,4,9,7,9}
        };
        for (int i=0; i<n; i++){
            for(int j=0; j<n ;j++){
                gsl_matrix_set(A, i, j, A_inside[i][j]);
        }
    }


    //std::cout << "Macierz A:" << std::endl;
    // MyFile<<"Macierz A:"<<std::endl;
    //print(A);
    gaus_jordan(A, b, x, answ);
    // std::cout << "Macierz A i wektor b po zmianach:" << std::endl;
    // MyFile<<"Macierz A i wektor b po zmianach:" <<std::endl;
    // print_answ(answ);
    multiply(A, x, c);
    std::cout << "VEctor b: \n";
    for (int i = 0; i < n; i++)
        std::cout << (double)gsl_vector_get(b, i) << " ";
    std::cout << std::endl;
    std::cout << "VEctor c: \n";
    for (int i = 0; i < n; i++)
        std::cout << (double)gsl_vector_get(c, i) << " ";
    std::cout << std::endl;
    std::cout<<q<<": "<<counting(b,c)<<std::endl;
    MyFile<<q<<": "<<counting(b,c)<<std::endl;

    }

    gsl_matrix_free(A);
    gsl_matrix_free(answ);
    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_vector_free(c);

    return 0;
}