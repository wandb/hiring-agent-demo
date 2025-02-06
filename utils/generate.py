import os
from fpdf import FPDF
import weave
import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI

from utils.prompt import SimpleApplicationGeneration, app_gen_prompt, app_gen_prompt_pos,app_gen_prompt_neg

# Function to generate a sample application
@weave.op
def generate_application(generation_model: str, job_offer: str, positive: bool):
    if positive:
        prompt = app_gen_prompt_pos.format(job_offer=job_offer)
        prompt += "\n The application should fit well and should likely be ACCEPTED."
    else:
        prompt = app_gen_prompt_neg.format(job_offer=job_offer)
        prompt += "\n The application should NOT fit well and should likely be NOT ACCEPTED."
    model = ChatOpenAI(
        model=generation_model,
        response_format={"type": "json"}).with_structured_output(SimpleApplicationGeneration)
    generated_application = model.invoke(prompt)
    return generated_application.application_text, generated_application.reason

# Function to save application as a PDF using fitz
@weave.op
def save_as_pdf(content, filename):
    """
    Save the given text content as a PDF file.

    Args:
        content (str): The text content to be saved as a PDF.
        filename (str): The name of the output PDF file.

    Returns:
        None
    """
    # Initialize FPDF object
    content =content[0]
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Use a Latin-1 compatible encoding, decode to ensure content is safely handled
    content = content.encode('latin-1', 'replace').decode('latin-1')

    # Set font and add content to the PDF
    pdf.set_font("Arial", size=12)

    # Add content to the PDF, splitting lines as needed
    lines = content.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, line)

    # Save the PDF
    pdf.output(filename)
    print(f"PDF saved as {filename}")

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    # Generate and save 5 sample applications
    from prepro import extract_text_from_pdf
    for file_name in os.listdir("./data/offers"):
        if file_name.endswith('.pdf'):
            print(file_name)
            pdf_path = f"data/offers/{file_name}"
            extracted_text = extract_text_from_pdf(pdf_path)
            print(extracted_text)
            for b in [True, False]:
                application_content = generate_application(generation_model="gpt-4o-mini", job_offer=extracted_text, positive=b)
                print(application_content)
                save_as_pdf(application_content, f"data/applications/sample_application_{file_name.split('.')[0]}_{b}.pdf")
            print("Sample applications have been generated and saved as PDFs.")