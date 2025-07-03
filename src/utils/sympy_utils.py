import sympy as sp
import numpy as np
import jax
import jax.numpy as jnp
import sys

def parse_function_input(str_func):
    """
    Parse a string input into a sympy symbolic expression.

    Args:
        str_func (str): The function as a string, e.g., "x**2 + y**2".

    Returns:
        sympy.Expr: The symbolic expression representing the function.
    """
    x, y = sp.symbols('x y')
    func = sp.sympify(str_func)
    return func, x, y

def find_critical_points(dfx, dfy):
    """
    Find critical points using either simple or full solving strategy.
    """
    critical_points = []

    # Sprawdź zależności zmiennych
    dfx_has_y = dfx.has(y)
    dfy_has_x = dfy.has(x)

    if not dfx_has_y and not dfy_has_x:
        # Prosty przypadek - rozdzielne równania
        sol_x = sp.solve(dfx, x)
        sol_y = sp.solve(dfy, y)

        # Tworzymy wszystkie kombinacje rozwiązań
        for x_val in sol_x:
            for y_val in sol_y:
                critical_points.append({x: x_val, y: y_val})
    else:
        # Przypadek ogólny - układ równań
        solutions = sp.solve([dfx, dfy], (x, y), dict=True)
        for sol in solutions:
            critical_points.append(sol)

    return critical_points

def compute_second_derivatives(dfx, dfy, critical_points):
    """
    Compute second order derivatives evaluated at critical points.

    Args:
        dfx (sympy.Expr): First derivative w.r.t. x
        dfy (sympy.Expr): First derivative w.r.t. y
        critical_points (list of dict): Points where derivatives are evaluated

    Returns:
        list of dict: Each dict contains the second derivatives at a point
    """
    df2 = []

    df_xx = sp.diff(dfx, x)
    df_xy = sp.diff(dfx, y)
    df_yy = sp.diff(dfy, y)
    df_yx = sp.diff(dfy, x)

    for pt in critical_points:
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
    return df2

def evaluate_hessian_at_point(df2):
    """
    Build Hessian matrices for each critical point.

    Args:
        df2 (list of dict): Output from compute_second_derivatives()

    Returns:
        list of dict: Each dict contains the point and its Hessian matrix
    """
    hessians = []
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
        hessians.append({
            "point": point,
            "hessian": H
        })
    return hessians

def classify_extremum(hessians):
    """
    Classify the type of extremum for each Hessian matrix.

    Args:
        hessians (list of dict): Output from evaluate_hessian_at_points()

    Returns:
        list of dict: Each dict includes point, Hessian, determinant and classification
    """
    results = []
    for key, value in enumerate(hessians, start = 1):
        point = value['point']

        detH = value['hessian'].det()

        H = value['hessian']

        if detH > 0:
            if H[0,0] > 0:
                classification = 'minimum'
            else:
                classification = 'maximum'
        elif detH <0:
                classification = 'saddle point'
        else:
                classification = 'undetermined'

        results.append({
            "point": point,
            "hessian": H,
            "determinant": detH,
            "classification": classification
        })
    return results