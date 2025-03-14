import numpy as np
import scipy.linalg 

def lu_decompose(A):
    P, L, U = np.linalg.lu(A)
    return L, U

def save_to_file(filename, data):
    with open(filename, 'w') as file:
        file.write(str(data))
        
def influence_of_diagonal(A):
    diag_elements = np.diag(A)
    return diag_elements

def inverse_matrix(A):
    A_inv = np.linalg.inv(A)
    return A_inv

def solve_system(A, b):
    x = np.linalg.solve(A, b)
    return x

def matrix_multiply(A, B):
    C = np.dot(A, B)
    return C

def condition_number(A):
    cond = np.linalg.cond(A)
    return cond

N = 4
# Przykład macierzy A
A = np.array([[1 / (i + j + 2) for i in range(N)] for j in range(N)])

P, L, U = scipy.linalg.lu(A)
for line in U:
    print(line)
# print("Macierz L:", L)
# print("Macierz U:", U)

diagonal_influence = influence_of_diagonal(U)
print("Elementy diagonalne:", diagonal_influence)

product_of_diagonal = np.prod(diagonal_influence)
print("Iloczyn elementów diagonalnych:", product_of_diagonal)

# Zapisz wynik do pliku
with open("results.txt", "w") as file:
    file.write("Elementy diagonalne: " + str(diagonal_influence) + "\n")
    file.write("Iloczyn elementów diagonalnych: " + str(product_of_diagonal) + "\n")

# Przykładowy wektor b
b = np.array([1, 1, 1, 1])
x = solve_system(A, b)
print("Rozwiązanie układu równań:", x)
 
A_inv = inverse_matrix(A)
print("Macierz odwrotna:", A_inv)
save_to_file("inverse_matrix.txt", A_inv)

C = matrix_multiply(A, A_inv)
save_to_file("matrix_c.txt", C)

# norm_A = np.max(np.abs(A))  # Maksymalny element w A
# norm_A_inv = np.max(np.abs(A_inv))  # Maksymalny element w A^-1

norm_A = np.linalg.cond(A, p=1)
norm_A_inv = np.linalg.cond(A_inv, p = np.inf)

# Wskaźnik uwarunkowania
cond = norm_A * norm_A_inv
print("Wskaźnik uwarunkowania macierzy:", norm_A_inv)
save_to_file("cond.txt", cond)

A_1 = matrix_multiply(L, U)
for l in A_1:
    print(l)
    
save_to_file("matrix_l.txt", L)
save_to_file("matrix_u.txt", U)