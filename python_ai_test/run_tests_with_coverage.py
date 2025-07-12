import subprocess
import os

def run_tests_with_coverage():
    test_file = "test_calculation_helper_full.py"
    html_report_dir = "htmlcov"

    # Verifica que coverage esté instalado
    try:
        subprocess.run(["coverage", "--version"], check=True)
    except subprocess.CalledProcessError:
        print("Error: coverage no está instalado. Instálalo con: pip install coverage")
        return
    except FileNotFoundError:
        print("Error: coverage no está instalado o no se encuentra en el PATH.")
        return

    # Ejecuta las pruebas y genera el reporte
    try:
        subprocess.run(["coverage", "run", test_file], check=True)
        subprocess.run(["coverage", "html", "-d", html_report_dir], check=True)
        print(f"✅ Reporte HTML generado en: {os.path.join(html_report_dir, 'index.html')}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando pruebas o generando el reporte: {e}")

if __name__ == "__main__":
    run_tests_with_coverage()
