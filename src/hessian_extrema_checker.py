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

sol_y = sp.solve(df_x, y) #Liczymy y z pochodnej czastkowej po x
print(f"Possible y values are: {sol_y}")

critical_points = []

for i in sol_y: #Tutaj robimy pętle, bo moze byc wiecej mozliwych wartosci y niz jedna
    df_y_sub = df_y.subs(y, i)
    sol_x = sp.solve(df_y_sub, x)

    for val_x in sol_x:
        val_y = i.subs(x, val_x)
        points = {x: val_x, y: val_y}
        critical_points.append(points)

df_xx = sp.diff(df_x, x)
df_xy = sp.diff(df_x, y)
df_yy = sp.diff(df_y, y)
df_yx = sp.diff(df_y, x)

df2 = []

#Wartosci pochodnych drugiego stopnia
for pt in critical_points:
    print("Punkt krytyczny: ", pt)
    v_xx = df_xx.subs(pt)
    v_yy = df_yy.subs(pt)
    v_xy = df_xy.subs(pt)
    v_yx = df_yx.subs(pt)
    df_points = {
        "points": pt,
        "df_xx": v_xx, 
        "df_yy": v_yy, 
        "df_xy": v_xy, 
        "df_yx": v_yx
    }
    df2.append(df_points)



#Walidacja symetrii pochodnych
if sp.simplify(df_xy - df_yx) != 0:  #sympy przechowuje je jako obiekty symboliczne, dlatego trzeba simplify
    print("It's incorrect, you cannot calculate hessian")
    sys.exit(1)

#Tworzenie macierzy Hesjana
for key, val in enumerate(df2, start = 1):
    point = val['points']

    #Wyciaganie wartosci
    v_xx = val['df_xx']
    v_yy = val['df_yy']
    v_xy = val['df_xy']
    v_yx = val['df_yx']

    H = sp.Matrix([
                [v_xx,v_xy],
                [v_yx,v_yy]
    ])

    print(f"Macierz {key} przypadku dla punktu {point}")
    sp.pprint(H)
    #Wyznacznik macierzy

    detH = H.det()
    print(detH)
    print(type(detH))

    #Sprawdzenie czy wystepuje ekstremum lokalne

    if detH > 0:
        if H[0,0] > 0:
            print('Funkcja ma ekstremum min')
        else:
            print('Funkcja ma ekstremum max')
    elif detH <0:
        print("funkcja nie ma ekstremum")
        sys.exit(1)
    else:
        print("Nie da sie okreslić czy występuje :)")
        sys.exit(1)

