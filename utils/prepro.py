import os
import weave
import fitz  # PyMuPDF
from PIL import Image
from typing import Union, Any

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
                text += page.get_text("text")
    except Exception as e:
        print(f"Error while reading the PDF: {e}")
    return text

@weave.op
def save_as_pdf(content, filename):
    # Unpack tuple if content is a tuple
    if isinstance(content, tuple):
        content = content[0]  # Take the first element (application_text)
        
    # Create a new PDF document
    pdf_document = fitz.open()
    # Add a new page
    page = pdf_document.new_page()
    # Define a font and text position
    font = "helv"  # Helvetica font
    font_size = 11  # Slightly smaller font
    margin = 72  # 1-inch margin
    y_position = margin
    max_width = page.rect.width - (2 * margin)  # Maximum line width
    
    # Create a font object
    font_obj = fitz.Font(fontname=font)
    
    # Write the content line by line
    for line in content.split("\n"):
        # Check if line needs to be wrapped
        text_width = font_obj.text_length(line, fontsize=font_size)
        if text_width > max_width:
            # Split into multiple lines
            words = line.split()
            current_line = []
            current_width = 0
            
            for word in words:
                word_width = font_obj.text_length(word + " ", fontsize=font_size)
                if current_width + word_width <= max_width:
                    current_line.append(word)
                    current_width += word_width
                else:
                    # Write current line and start new one
                    if y_position + font_size > page.rect.height - margin:
                        page = pdf_document.new_page()
                        y_position = margin
                    page.insert_text((margin, y_position), " ".join(current_line), fontsize=font_size, fontname=font)
                    y_position += font_size + 4
                    current_line = [word]
                    current_width = word_width
            
            # Write remaining words
            if current_line:
                if y_position + font_size > page.rect.height - margin:
                    page = pdf_document.new_page()
                    y_position = margin
                page.insert_text((margin, y_position), " ".join(current_line), fontsize=font_size, fontname=font)
                y_position += font_size + 4
        else:
            # Write single line
            if y_position + font_size > page.rect.height - margin:
                page = pdf_document.new_page()
                y_position = margin
            page.insert_text((margin, y_position), line, fontsize=font_size, fontname=font)
            y_position += font_size + 4
    
    # Save the PDF document
    pdf_document.save(filename)
    pdf_document.close()

@weave.op
def pre_process_eval(row: Any):
    return {
        "offer_pdf": str(row["offer_pdf"]),
        "offer_text": row["offer_text"],
        "application_pdf": str(row["application_pdf"]),
        "application_text": row["application_text"],
        "interview": row["interview"],
        "reason": row["reason"],
        "offer_images": pdf_to_images(str(row["offer_pdf"])),
        "application_images": pdf_to_images(str(row["application_pdf"])),
    }

if __name__ == "__main__":
    # Example usage
    pdf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "/Users/niware_wb/Documents/wandb_internal/hiring-agent-demo/utils/data/offers/Financial_Data_Analyst.pdf"
    )
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)
