import sympy as sp
import numpy as np
import jax
import jax.numpy as jnp
import sys

x, y = sp.symbols('x y')
input_str = input("Enter the function in terms of x and y (e.g., 'x**2 + y**2'): ")
f = sp.sympify(input_str) #Konvert input string to sympy expression
print("\n Your function is:\n")
sp.pprint(f)

df_x = sp.diff(f, x)
df_y = sp.diff(f, y) 

solution = sp.solve([df_x, df_y], (x, y)) 
print(solution)

df_xx = sp.diff(df_x, x)
df_xy = sp.diff(df_x, y)
df_yy = sp.diff(df_y, y)
df_yx = sp.diff(df_y, x)


#Walidacja symetrii pochodnych
if sp.simplify(df_xy - df_yx) != 0:  #sympy przechowuje je jako obiekty symboliczne, dlatego trzeba simplify
    print("It's incorrect, you cannot calculate hessian")
    sys.exit(1)

#Tworzenie macierzy Hesjana

A = sp.Matrix([
            [sp.simplify(df_xx),sp.simplify(df_xy)],
            [sp.simplify(df_yx),sp.simplify(df_yy)]
])

print(A)
#Wyznacznik macierzy

detA = A.det()
print(detA)
print(type(detA))

#Sprawdzenie czy wystepuje ekstremum lokalne

if detA > 0:
    if A[0,0] > 0:
        print('Funkcja ma ekstremum min')
    else:
        print('Funkcja ma ekstremum max')
elif detA <0:
    print("funkcja nie ma ekstremum")
    sys.exit(1)
else:
    print("Nie da sie okreslić czy występuje")
    sys.exit(1)

