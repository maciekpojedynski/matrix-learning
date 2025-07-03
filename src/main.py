import sympy as sp
import numpy as np
import jax
import jax.numpy as jnp
import sys
from utils.sympy_utils import (
    parse_function_input,
    find_critical_points, 
    compute_second_derivatives, 
    evaluate_hessian_at_point, 
    classify_extremum
)
from utils.export_utils import export_to_postgres

input_str = input("Enter the function in terms of x and y (e.g., 'x**2 + y**2'): ")

f, x, y = parse_function_input(input_str)

df_x = sp.diff(f, x)
df_y = sp.diff(f, y)

critical_points = find_critical_points(df_x, df_y, x, y)
second_derivatives = compute_second_derivatives(df_x, df_y, critical_points, x, y)
Hessians = evaluate_hessian_at_point(second_derivatives)
classified = classify_extremum(Hessians)

for idx, item in enumerate(classified, start=1):
    print(f"\n=== Punkt krytyczny {idx} ===")
    print(f"Point: {item['point']}")
    sp.pprint(item['hessian'])
    print(f"Determinant: {item['determinant']}")
    print(f"Classification: {item['classification']}")

export_to_postgres( 
    classified, 
    'Extrema', 
    'postgres', 
    'maciek2102', 
    host='localhost', 
    port="5432")