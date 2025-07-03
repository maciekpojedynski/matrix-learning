# Matrix Learning – Hessian Extrema Calculator

This project is a Python tool that analyzes mathematical functions of two variables, computes critical points, builds the Hessian matrix, and classifies local extrema (minimum, maximum, saddle point).

The application also supports batch processing of multiple functions from CSV files and exporting results to a PostgreSQL database.Repozytorium z kodami i notatkami do nauki NumPy i operacji na macierzach.

## 📂 Project Structure

notebooks/ – example Jupyter notebooks for code exploration and experiments.
src/ – target, organized Python modules containing functions and classes.
tests/ – unit tests (e.g., for pytest).
data/ – (optional) input data or sample CSV files.
README.md – project description.
requirements.txt – list of Python dependencies.

## 📚 Features

✅ Compute first and second derivatives (using SymPy)  
✅ Detect critical points (solving equations and systems)  
✅ Build Hessian matrices  
✅ Classify extrema based on the determinant and leading minor  
✅ Filter out non-real solutions  
✅ Batch processing of multiple functions from CSV  
✅ Export results to PostgreSQL  
✅ Modular project structure (utils, main pipeline, data)

## 🛠️ Technologies

- **Python 3**
- [SymPy](https://www.sympy.org)
- [Pandas](https://pandas.pydata.org/)
- [Psycopg2](https://www.psycopg.org/)
- [NumPy](https://numpy.org/)
- PostgreSQL

## 🚀 Project Structure

matrix-learning/
├── src/
│ ├── main.py # Main execution script
│ ├── utils/
│ │ ├── sympy_utils.py # Functions for symbolic computations
│ │ ├── export_utils.py # Export to PostgreSQL
│ │ └── csv_processor.py # Batch processing from CSV
│ └── data/
│ └── functions.csv # Example input data
├── notebooks/
│ └── explorations.ipynb # Experimentation / drafts
├── README.md

## 📝 Usage

### Single Function Calculation

Run `main.py` and provide a function input when prompted:

```bash
python src/main.py

x**2 + y**2 + x*y

Batch Processing from CSV

Provide a CSV file in src/data/functions.csv with functions in the function column:

function
x**2 + y**2
x**3 + y**3 - 3*x*y
x*y + log(x+1) + exp(y)


Run csv_processor.py (or call process_functions_from_csv() from main.py) to process all functions and export results to PostgreSQL.

## 💾 Exporting to PostgreSQL

Before running, create a database called `Extrema` (or adjust `dbname` in the script).

Example SQL to create the database:

```sql
CREATE DATABASE Extrema;

Credentials and connection parameters can be set in main.py or csv_processor.py.

Results will be inserted into a table called wyniki.


