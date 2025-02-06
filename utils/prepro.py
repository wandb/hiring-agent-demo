import weave
import fitz  # PyMuPDF
from PIL import Image
from typing import Union

@weave.op
def pdf_to_images(pdf_path: str, single_img: bool = True) -> Union[list[Image.Image], Image.Image]:
    """Convert PDF pages to PIL Images.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of PIL Images, one per page
    """
    pdf_document = fitz.open(pdf_path)
    images = []
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        
    pdf_document.close()

    # NOTE: return first image for better demo purposes (to see img in trace-table-view)
    return images[0] if single_img else images

@weave.op
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception as e:
        print(f"Error while reading the PDF: {e}")
    return text

if __name__ == "__main__":
    # Example usage
    pdf_path = "data/offers/Werkstudent_Aushilfe Kundenservice Krankenversicherung (m_w_d).pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)
