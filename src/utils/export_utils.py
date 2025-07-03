import psycopg2 
import sympy as sp


def export_to_postgres(results, dbname, user, password, host='localhost', port="5432"):
    """
    Exports a list of results dictionary to PostgreSQL.
    """

    conn = psycopg2.connect(
    dbname = dbname,
    user = user,
    password = password,
    host = host,
    port = port
    )

    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS wyniki (
    id SERIAL PRIMARY KEY,
    x_value DOUBLE PRECISION,
    y_value DOUBLE PRECISION,
    determinant DOUBLE PRECISION,
    classification TEXT
    )
    """)

    for item in results:
        point = item["point"]
        print("DEBUG point:", point, type(point))

        if isinstance(point, dict):
            x_value = point[sp.Symbol("x")]
            y_value = point[sp.Symbol("y")]
        elif isinstance(point, (tuple, list)):
            if len(point) == 2:
                x_value = point[0]
                y_value = point[1]
            elif len(point) == 1:
                x_value = point[0]
                y_value = None
            else:
                raise ValueError(f"Unknown tuple length: {len(point)}")
        elif isinstance(point, (int, float)):
            x_value = point
            y_value = None
        else:
            raise ValueError(f"Unknown point format: {point}")
        
        x_value = float(x_value)
        y_value = float(y_value) if y_value is not None else None
        det = float(item["determinant"])
        clas = str(item["classification"])
        
        det = float(item['determinant'])
        clas = item['classification']

        print("DEBUG INSERT VALUES:", (x_value, y_value, det, clas))

        cur.execute(
            """
            INSERT INTO wyniki (x_value, y_value, determinant, classification)
            VALUES (%s, %s, %s, %s)
            """,
            (x_value, y_value, det, clas)
        )

    conn.commit()
    cur.close()
    conn.close()
