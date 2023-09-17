import os
import requests
import zipfile

# List of file names without paths
file_names = [
    "truncated_260_to_284.xlsx_vectorizer.pkl",
    "not_trancated_full_paragraph.xlsx_extra_trees_model.pkl",
    "not_trancated_full_paragraph.xlsx_ridge_model.pkl",
    "not_trancated_full_paragraph.xlsx_vectorizer.pkl",
    "truncated_10_to_34.xlsx_extra_trees_model.pkl",
    "truncated_10_to_34.xlsx_ridge_model.pkl",
    "truncated_10_to_34.xlsx_vectorizer.pkl",
    "truncated_35_to_59.xlsx_extra_trees_model.pkl",
    "truncated_35_to_59.xlsx_ridge_model.pkl",
    "truncated_35_to_59.xlsx_vectorizer.pkl",
    "truncated_60_to_84.xlsx_extra_trees_model.pkl",
    "truncated_60_to_84.xlsx_ridge_model.pkl",
    "truncated_60_to_84.xlsx_vectorizer.pkl",
    "truncated_85_to_109.xlsx_extra_trees_model.pkl",
    "truncated_85_to_109.xlsx_ridge_model.pkl",
    "truncated_85_to_109.xlsx_vectorizer.pkl",
    "truncated_110_to_134.xlsx_extra_trees_model.pkl",
    "truncated_110_to_134.xlsx_ridge_model.pkl",
    "truncated_110_to_134.xlsx_vectorizer.pkl",
    "truncated_135_to_159.xlsx_extra_trees_model.pkl",
    "truncated_135_to_159.xlsx_ridge_model.pkl",
    "truncated_135_to_159.xlsx_vectorizer.pkl",
    "truncated_160_to_184.xlsx_extra_trees_model.pkl",
    "truncated_160_to_184.xlsx_ridge_model.pkl",
    "truncated_160_to_184.xlsx_vectorizer.pkl",
    "truncated_185_to_209.xlsx_extra_trees_model.pkl",
    "truncated_185_to_209.xlsx_ridge_model.pkl",
    "truncated_185_to_209.xlsx_vectorizer.pkl",
    "truncated_210_to_234.xlsx_extra_trees_model.pkl",
    "truncated_210_to_234.xlsx_ridge_model.pkl",
    "truncated_210_to_234.xlsx_vectorizer.pkl",
    "truncated_235_to_259.xlsx_extra_trees_model.pkl",
    "truncated_235_to_259.xlsx_ridge_model.pkl",
    "truncated_235_to_259.xlsx_vectorizer.pkl",
    "truncated_260_to_284.xlsx_extra_trees_model.pkl",
    "truncated_260_to_284.xlsx_ridge_model.pkl"
]

def check_and_download_files():
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
            url = 'URL_TO_DOWNLOAD_CONTENT.ZIP'  # Replace with the actual URL

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

# Uncomment the line below if you want the check_and_download_files function to run when this module is executed directly.
# check_and_download_files()
