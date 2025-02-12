import os
import weave
import fitz  # PyMuPDF
from typing import List
from langchain_openai import ChatOpenAI

# remove utils for __main__ test
from utils.prompt import (
    app_gen_prompt_pos, app_gen_prompt_neg,
    SimpleApplicationGeneration, EvaluationDataset, EvaluationExample)
from utils.prepro import extract_text_from_pdf, save_as_pdf

@weave.op
def generate_application(generation_model: str, job_offer: str, positive: bool):
    if positive:
        prompt = app_gen_prompt_pos.format(job_offer=job_offer)
    else:
        prompt = app_gen_prompt_neg.format(job_offer=job_offer)
    model = ChatOpenAI(
        model=generation_model,
        response_format={"type": "json"}).with_structured_output(SimpleApplicationGeneration)
    generated_application = model.invoke(prompt)
    return generated_application.application_text, generated_application.reason

@weave.op()
def generate_dataset(offer_list: List[str], generation_model: str, num_app: int) -> EvaluationDataset:    
    examples = []
    for offer_path in offer_list:
        # Extract text from offer PDF
        offer_text = extract_text_from_pdf(offer_path)
        offer_id = os.path.basename(offer_path).split(".")[0]
        print("offer_id: ", offer_id)
        # Generate positive and negative applications
        for i in range(num_app):
            # Generate positive application
            pos_app_content, pos_reasoning = generate_application(
                generation_model=generation_model, job_offer=offer_text, positive=True)
            pos_app_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "data/applications", 
                f"sample_application_pos_{i}_{offer_id}.pdf"
            ))
            save_as_pdf(pos_app_content, pos_app_path)
            
            # Generate negative application (with different qualifications)
            neg_app_content, neg_reasoning = generate_application(
                generation_model=generation_model, job_offer=offer_text, positive=False)
            neg_app_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                "data/applications",
                f"sample_application_neg_{i}_{offer_id}.pdf"
            ))
            save_as_pdf(neg_app_content, neg_app_path)
            
            # Add positive example
            examples.append(EvaluationExample(
                offer_pdf=os.path.abspath(offer_path),
                offer_text=offer_text,
                application_pdf=pos_app_path,
                application_text=pos_app_content,
                interview=True,
                reason=pos_reasoning
            ))
            
            # Add negative example  
            examples.append(EvaluationExample(
                offer_pdf=os.path.abspath(offer_path),
                offer_text=offer_text,
                application_pdf=neg_app_path,
                application_text=neg_app_content,
                interview=False,
                reason=neg_reasoning
            ))
    return EvaluationDataset(examples=examples)


if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv

    # Add the parent directory to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    load_dotenv("utils/.env")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    # Generate and save 2 sample applications (pos, neg) per offer
    current_dir = os.path.dirname(os.path.abspath(__file__))
    offers_dir = os.path.join(current_dir, "data", "offers")
    applications_dir = os.path.join(current_dir, "data", "applications")
    
    for file_name in os.listdir(offers_dir):
        if file_name.endswith('.pdf'):
            print(f"File Name:\n#######\n {file_name}\n#######\n\n")

            pdf_path = os.path.join(offers_dir, file_name)
            extracted_text = extract_text_from_pdf(pdf_path)
            print(f"Extracted Text:\n#######\n {extracted_text}\n#######\n\n")

            for b in [True, False]:
                output_path = os.path.join(applications_dir, f"sample_application_{file_name.split('.')[0]}_{b}.pdf")
                if os.path.exists(output_path):
                    print(f"Warning: Application file {output_path} already exists and will be overwritten.")
                
                application_content = generate_application(generation_model="gpt-4o-mini", job_offer=extracted_text, positive=b)
                print(f"Application Content:\n#######\n {application_content}\n#######\n\n")
                
                save_as_pdf(application_content, output_path)
            print("Sample applications have been generated and saved as PDFs.")