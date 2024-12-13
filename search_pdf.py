import os
from PyPDF2 import PdfReader

def search_string_in_pdf(directory, search_string):
    """
    Searches for a specific string in all PDF files in a given directory.

    :param directory: The directory containing the PDF files.
    :param search_string: The string to search for.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    pdf_files = [file for file in os.listdir(directory) if file.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in directory '{directory}'.")
        return

    print(f"Searching for '{search_string}' in PDF files in '{directory}'...\n")

    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        try:
            reader = PdfReader(file_path)
            found = False

            for page_number, page in enumerate(reader.pages, start=1):
                if search_string in page.extract_text():
                    print(f"Found '{search_string}' in file '{pdf_file}', page {page_number}.")
                    found = True
                    break  # Stop searching further in this file

            if not found:
                print(f"'{search_string}' not found in file '{pdf_file}'.")
        except Exception as e:
            print(f"Error reading '{pdf_file}': {e}")

# Example usage
# directory_path = input("Enter the directory path: ")
directory_path = "C:\\Users\\Dell\\OneDrive\\stock.topicbin.com\\Amity\\MCA\\Sem IV\\Major Project\\NEPSE Index Forecasting"
search_text = input("Enter the string to search: ")
search_string_in_pdf(directory_path, search_text)
