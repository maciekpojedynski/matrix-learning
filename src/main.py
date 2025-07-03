from utils.csv_processor import process_functions_from_csv

load = input("Load functions from CSV and process? (y/n): ")
if load.lower() == "y":
    process_functions_from_csv(
        csv_path="src/data/functions.csv",
        dbname="Extrema",
        user="postgres",
        password="twoje_haslo",
        host="localhost"
    )