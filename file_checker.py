import os
import requests
import zipfile

def check_and_download_files(file_names):
    missing_files = []
    for file_name in file_names:
        if not os.path.exists(file_name):
            missing_files.append(file_name)

    if missing_files:
        print("The following files are missing:")
        for file_name in missing_files:
            print(file_name)

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            }
            url = 'https://mtc.best/content.zip'  # Replace with the actual URL

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            with open('content.zip', 'wb') as zip_file:
                zip_file.write(response.content)

            with zipfile.ZipFile('content.zip', 'r') as zip_ref:
                zip_ref.extractall()

            print("content.zip downloaded and extracted successfully.")
        except Exception as e:
            print(f"Error downloading or extracting content.zip: {e}")
    else:
        print("All files exist.")
