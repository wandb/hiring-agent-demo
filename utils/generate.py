import os
import weave
import fitz  # PyMuPDF
from typing import List
from langchain_openai import ChatOpenAI

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import wandb
import json

# remove utils for __main__ test
from utils.prompt import (
    app_gen_prompt_pos, app_gen_prompt_neg,
    SimpleApplicationGeneration, EvaluationDataset, EvaluationExample)
from utils.prepro import extract_text_from_pdf, save_as_pdf

@weave.op
def generate_applicant_characteristics(num_applicants: int, job_positions: List[str], 
                                       bias_factors: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Generate a structured table with representative characteristics of hypothetical applicants.
    
    Args:
        num_applicants: Number of applicants to generate
        job_positions: List of job position IDs to distribute applicants across
        bias_factors: Optional dictionary of bias factors to introduce in the dataset
    
    Returns:
        DataFrame with applicant characteristics
    """
    # Default bias factors if none provided
    if bias_factors is None:
        bias_factors = {
            "gender": 0.0,  # 0.0 means no bias, positive values bias toward male
            "age": 0.0,     # 0.0 means no bias, positive values bias toward younger
            "nationality": 0.0  # 0.0 means no bias, positive values bias toward certain nationalities
        }
    
    # Generate basic characteristics
    data = []
    for i in range(num_applicants):
        # Assign to a random job position
        job_position = np.random.choice(job_positions)
        
        # Generate gender with potential bias
        gender_prob = 0.5 + (bias_factors.get("gender", 0) * 0.5)  # Convert bias to probability
        gender = "Male" if np.random.random() < gender_prob else "Female"
        
        # Generate age with potential bias (25-60 range)
        age_bias = bias_factors.get("age", 0) * 10  # Scale bias factor
        age = max(25, min(60, int(np.random.normal(40 - age_bias, 10))))
        
        # Generate nationality (simplified list)
        nationalities = ["US", "UK", "Germany", "France", "China", "India", "Brazil", "Nigeria"]
        # Apply nationality bias if present
        if bias_factors.get("nationality", 0) > 0:
            # Bias toward first few nationalities
            nationality_probs = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
        elif bias_factors.get("nationality", 0) < 0:
            # Bias toward last few nationalities
            nationality_probs = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.2, 0.3])
        else:
            # No bias - equal probability
            nationality_probs = np.ones(len(nationalities)) / len(nationalities)
            
        nationality = np.random.choice(nationalities, p=nationality_probs)
        
        # Generate education level
        education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
        education_probs = [0.1, 0.5, 0.3, 0.1]
        education = np.random.choice(education_levels, p=education_probs)
        
        # Generate years of experience (correlated with age)
        max_experience = max(0, age - 22)  # Assuming education completed by 22
        years_experience = max(0, min(max_experience, int(np.random.normal(max_experience * 0.7, 5))))
        
        # Generate a quality score (0-100)
        # This is a simplified score that would be calculated based on job requirements
        base_quality = np.random.normal(70, 15)
        
        # Adjust quality based on experience (more experience generally means higher quality)
        experience_factor = min(1.0, years_experience / 10) * 10  # Up to 10 points for experience
        
        # Adjust quality based on education
        education_bonus = {"High School": 0, "Bachelor's": 5, "Master's": 10, "PhD": 15}
        education_factor = education_bonus[education]
        
        # Calculate final quality score
        quality_score = min(100, max(0, base_quality + experience_factor + education_factor))
        
        # Determine if this is a positive or negative example
        # Higher quality scores are more likely to be positive examples
        is_positive = quality_score >= 70  # Threshold for positive example
        
        data.append({
            "job_position": str(job_position),
            "gender": gender,
            "age": age,
            "nationality": str(nationality),
            "education": str(education),
            "years_experience": years_experience,
            "quality_score": quality_score,
            "is_positive": is_positive
        })
    
    return pd.DataFrame(data)

@weave.op
def calculate_r_score(df: pd.DataFrame) -> float:
    """
    Calculate the R score (representativeness) for the dataset.
    
    This is a simplified implementation of the R score based on ISO standards.
    It evaluates representativeness, completeness, and balance of the dataset.
    
    Args:
        df: DataFrame with applicant characteristics
        
    Returns:
        R score between 0 and 1, where higher is better
    """
    # Check completeness (no missing values)
    completeness = 1.0 - (df.isna().sum().sum() / (df.shape[0] * df.shape[1]))
    
    # Check gender balance (should be close to 50/50)
    gender_balance = 1.0 - abs(df['gender'].value_counts(normalize=True).get('Male', 0) - 0.5) / 0.5
    
    # Check age distribution (should cover the full range)
    age_range = df['age'].max() - df['age'].min()
    age_coverage = min(1.0, age_range / 35)  # Expecting range of at least 35 years (25-60)
    
    # Check nationality distribution (should have multiple nationalities)
    nationality_count = df['nationality'].nunique()
    nationality_diversity = min(1.0, nationality_count / 5)  # Expecting at least 5 nationalities
    
    # Check job position distribution (should be balanced across positions)
    position_counts = df['job_position'].value_counts(normalize=True)
    expected_freq = 1.0 / position_counts.shape[0]
    position_balance = 1.0 - position_counts.apply(lambda x: abs(x - expected_freq)).mean() / expected_freq
    
    # Check positive/negative example balance (should be somewhat balanced)
    positive_ratio = df['is_positive'].mean()
    example_balance = 1.0 - abs(positive_ratio - 0.5) / 0.5
    
    # Calculate overall R score (weighted average)
    r_score = (
        0.2 * completeness +
        0.15 * gender_balance +
        0.15 * age_coverage +
        0.15 * nationality_diversity +
        0.15 * position_balance +
        0.2 * example_balance
    )
    
    return r_score

@weave.op
def generate_application_from_characteristics(
    characteristics_df: pd.DataFrame,
    offer_paths: Dict[str, str],
    generation_model: str
) -> EvaluationDataset:
    """
    Generate application PDFs based on the characteristics table.
    
    Args:
        characteristics_df: DataFrame with applicant characteristics
        offer_paths: Dictionary mapping job position IDs to offer PDF paths
        generation_model: Model to use for generation
        
    Returns:
        EvaluationDataset with generated applications
    """
    examples = []
    
    for _, row in characteristics_df.iterrows():
        job_position = row['job_position']
        offer_path = offer_paths.get(job_position)
        
        if not offer_path:
            print(f"Warning: No offer path found for job position {job_position}")
            continue
            
        # Extract text from offer PDF
        offer_text = extract_text_from_pdf(offer_path)
        offer_id = os.path.basename(offer_path).split(".")[0]
        
        # Create a prompt context with the applicant characteristics
        applicant_context = f"""
        Applicant Profile:
        - Gender: {row['gender']}
        - Age: {row['age']}
        - Nationality: {row['nationality']}
        - Education: {row['education']}
        - Years of Experience: {row['years_experience']}
        - Quality Score: {row['quality_score']}
        """
        
        # Generate application based on characteristics and whether it should be positive or negative
        is_positive = row['is_positive']
        
        app_content, app_reasoning = generate_application(
            generation_model=generation_model,
            job_offer=offer_text,
            positive=is_positive,
            applicant_context=applicant_context  # Pass the context to the generation function
        )
        
        # Create a unique ID for this application
        app_id = f"{'pos' if is_positive else 'neg'}_{job_position}_{row.name}"
        
        # Save the application as PDF
        app_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "data/applications",
            f"application_{app_id}.pdf"
        ))
        save_as_pdf(app_content, app_path)
        
        # Add example to the dataset
        examples.append(EvaluationExample(
            offer_pdf=os.path.abspath(offer_path),
            offer_text=offer_text,
            application_pdf=app_path,
            application_text=app_content,
            interview=is_positive,
            reason=app_reasoning
        ))
    
    return EvaluationDataset(examples=examples)

# Update the generate_application function to accept applicant context
@weave.op
def generate_application(generation_model: str, job_offer: str, positive: bool, applicant_context: str = ""):
    if positive:
        prompt = app_gen_prompt_pos.format(job_offer=job_offer, applicant_context=applicant_context)
    else:
        prompt = app_gen_prompt_neg.format(job_offer=job_offer, applicant_context=applicant_context)
    model = ChatOpenAI(
        model=generation_model,
        response_format={"type": "json"}).with_structured_output(SimpleApplicationGeneration)
    generated_application = model.invoke(prompt)
    return generated_application.application_text, generated_application.reason

# @weave.op
# def generate_application(generation_model: str, job_offer: str, positive: bool):
#     if positive:
#         prompt = app_gen_prompt_pos.format(job_offer=job_offer)
#     else:
#         prompt = app_gen_prompt_neg.format(job_offer=job_offer)
#     model = ChatOpenAI(
#         model=generation_model,
#         response_format={"type": "json"}).with_structured_output(SimpleApplicationGeneration)
#     generated_application = model.invoke(prompt)
#     return generated_application.application_text, generated_application.reason

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