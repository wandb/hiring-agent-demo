import os
import weave
import fitz  # PyMuPDF
from PIL import Image
from typing import Union, Any, Callable, TypeVar
import io
import time
import functools

# Create a simple retry decorator for parallel Waeve evals
# Define a return type variable for the decorator
T = TypeVar('T')

def simple_retry(max_attempts: int = 2, delay: float = 1.0) -> Callable:
    """A simple retry decorator with fixed delay for file operations.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 2)
        delay: Delay between retries in seconds (default: 1.0)
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except OSError as e:
                    # Handle file limit errors
                    if e.errno == 24 and attempt < max_attempts - 1:
                        print(f"Too many open files. Waiting {delay}s before retry {attempt+1}/{max_attempts}")
                        time.sleep(delay)
                    else:
                        raise
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(f"Error in {func.__name__}: {str(e)}. Retrying in {delay}s ({attempt+1}/{max_attempts})")
                        time.sleep(delay)
                    else:
                        print(f"Failed after {max_attempts} attempts: {str(e)}")
                        raise
            # This should never be reached due to the raise in the last iteration,
            # but it's here for type checking completeness
            raise RuntimeError(f"Unexpected end of retry loop in {func.__name__}")
        return wrapper
    return decorator

@weave.op
@simple_retry(max_attempts=2, delay=1.0)
def pdf_to_images(pdf_path: str, single_img: bool = True) -> Union[list[Image.Image], Image.Image]:
    """Convert PDF pages to PIL Images.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of PIL Images, one per page
    """
    images = []
    
    try:
        with fitz.open(pdf_path) as pdf_document:
            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                pixmap = page.get_pixmap()
                img_data = pixmap.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
        # NOTE: return first image for better demo purposes (to see img in trace-table-view)
        return images[0] if single_img else images
    except Exception as e:
        print(f"Failed to process PDF: {e}")
        # Return empty list rather than raising exception for robustness
        return []

@weave.op
@simple_retry(max_attempts=2, delay=1.0)
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
@simple_retry(max_attempts=2, delay=1.0)
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
