from flask import Flask, jsonify, request
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import re
import time
import os
import zipfile
import shutil

app = Flask(__name__)
CORS(app)

download_path = os.path.join(os.path.expanduser("~"), "Downloads")
extracted_files_path = os.path.join(os.getcwd(), "extracted_files")

if not os.path.exists(extracted_files_path):
    os.makedirs(extracted_files_path)

def extract_files(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def move_files(extract_from, move_to):
    moved_files = []
    for root, dirs, files in os.walk(extract_from):
        for file in files:
            if file.endswith(('.csv', '.txt', '.xlsx')):
                shutil.move(os.path.join(root, file), os.path.join(move_to, file))
                moved_files.append(file)
    return moved_files

@app.route('/scrape', methods=['GET'])
def scrape():
    files_moved = []
    try:
        driver = webdriver.Chrome()
        driver.get("https://guiadevalores.fasecolda.com/ConsultaExplorador/Default.aspx?url=C:/inetpub/wwwroot/Fasecolda/ConsultaExplorador/Guias/GuiaValores_NuevoFormato")
        time.sleep(5)

        links = driver.find_elements(By.XPATH, "//td/a[contains(@id, 'gv_data_btn_archivo_')]")
        max_version = -1
        latest_link = None

        for link in links:
            match = re.search(r'gv_data_btn_archivo_(\d+)', link.get_attribute('id'))
            if match:
                version = int(match.group(1))
                if version > max_version:
                    max_version = version
                    latest_link = link

        if latest_link:
            driver.execute_script("arguments[0].scrollIntoView(true);", latest_link)
            time.sleep(1)

            actions = ActionChains(driver)
            actions.move_to_element(latest_link).click().perform()
            time.sleep(5)

            archivos = driver.find_elements(By.XPATH, "//td/a[contains(@id, 'gv_data_btn_archivo_')]")
            archivos_nombres = {}

            for archivo in archivos:
                archivo_id = archivo.get_attribute("id")
                archivo_nombre = archivo.text
                archivos_nombres[archivo_id] = archivo_nombre

            archivos_predeterminados = [
                "gv_data_btn_archivo_0",
                "gv_data_btn_archivo_2",
                "gv_data_btn_archivo_6",
                "gv_data_btn_archivo_9",
                "gv_data_btn_archivo_13"
            ]

            for archivo_id in archivos_predeterminados:
                if archivo_id in archivos_nombres:
                    archivo = driver.find_element(By.ID, archivo_id)
                    driver.execute_script("arguments[0].scrollIntoView(true);", archivo)
                    time.sleep(1)
                    archivo.click()
                    time.sleep(1)

            time.sleep(20)

            for file in os.listdir(download_path):
                if file.endswith('.zip'):
                    zip_path = os.path.join(download_path, file)
                    extract_files(zip_path, extracted_files_path)

            files_moved = move_files(extracted_files_path, extracted_files_path)

        driver.quit()

        return jsonify({"status": "success", "files_moved": files_moved})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
