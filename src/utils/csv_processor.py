import psycopg2 
import sympy as sp
import pandas as pd
from utils.sympy_utils import (
    parse_function_input,
    find_critical_points,
    compute_second_derivatives,
    evaluate_hessian_at_point,
    classify_extremum
)
from utils.export_utils import export_to_postgres

def process_functions_from_csv(
    csv_path,
    dbname,
    user,
    password,
    host="localhost",
    port="5432"
):
    df = pd.read_csv(csv_path)
    all_results = []

    for idx, row in df.iterrows():
        func_str = row["function"]
        print(f"\nProcessing function: {func_str}")

        f, x, y = parse_function_input(func_str)
        df_x = sp.diff(f, x)
        df_y = sp.diff(f, y)

        critical_points = find_critical_points(df_x, df_y, x, y)
        second_derivatives = compute_second_derivatives(df_x, df_y, critical_points, x, y)
        Hessians = evaluate_hessian_at_point(second_derivatives)
        classified = classify_extremum(Hessians)

        all_results.extend(classified)

    export_to_postgres(
        results=all_results,
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )