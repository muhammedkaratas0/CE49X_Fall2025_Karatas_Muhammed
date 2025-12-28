from pypdf import PdfReader

def extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print("--- PDF CONTENT START ---")
        print(text)
        print("--- PDF CONTENT END ---")
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    extract_text("Final_Project.pdf")
