import os
import wandb
import json
import weave
from weave.trace import urls
from openai import OpenAI
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import concurrent.futures
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv(".env")

# Initialize OpenAI client
openai_client = OpenAI()

# Number of parallel threads to use
NUM_THREADS = 10

# Define data models based on the existing ones in the codebase
class EvaluationExample(BaseModel):
    """Class representing a single evaluation example for testing the hiring agent."""
    offer_pdf: str = Field(description="Path to the job offer PDF file")
    offer_text: str = Field(description="Extracted text content from the job offer")
    application_pdf: str = Field(description="Path to the application PDF file") 
    application_text: str = Field(description="Extracted text content from the application")
    interview: bool = Field(description="Expected interview decision (True/False)")
    reason: str = Field(description="Expected reasoning for the interview decision")

class EvaluationDataset(BaseModel):
    """Class representing a collection of evaluation examples."""
    examples: List[EvaluationExample] = Field(description="List of evaluation examples for testing")

# The reason_comp_prompt from prompt.py
reason_comp_prompt = """You are a Senior Hiring Manager at Weights & Biases. Your task is to evaluate the decision rationale provided by a Junior Hiring Manager about inviting a candidate to interview. You have:

- A **reference hiring reason** (reason given by another senior hiring manager to guide your evaluation)  
- A **junior hiring reason** (reason given  by the junior hiring manager)  

First, use the following **Explainable 1-5 Rating Scale** to score the hiring reason that the junior hiring manager gave for their decision:

5 - Outstanding: Exemplary, comprehensive, perfectly aligned with W&B's hiring standards.
4 - Exceeds Standards: Strong, well-structured, minor omissions only.  
3 - Meets Standards: Acceptable, covers basics but lacks depth.  
2 - Below Standards: Superficial or incomplete, missing critical aspects.  
1 - Unsatisfactory: Fails to address core criteria, includes irrelevant or biased reasoning.  

Then, score the hiring reason provided by the junior hiring manager based on **five specific metrics**:

1. Position Fit Analysis - did the junior manager validate whether the candidate's location, salary expectations, and general profile match the job description?
2. Experience Analysis - did the junior hiring manager validate whether the experience of the candidate matches the job requirements?
3. Values Alignment  - did the junior hiring manager validate whether the candidate is aligned with the values of Weights & Biases (Honesty, Curiosity, Gumption, Grit)? 
4. Provided Evidence - did the junior hiring manager provide evidence for each of the statements in the hiring decision?
5. Fairness & Objectivity  - did the junior hiring manager only make unbiased arguments in their decision?

For each metric, output:
- **Score**: integer 1-5  
- **Comment**: brief justification citing evidence from the junior hiring manager's provided reason

Finally, provide an overall **Pass/Fail** recommendation on whether the reason provided by the junior hiring manager is valid to support the hiring decision. Be strict on whether the given hiring reason lives up to our hiring standards by performing well on the metrics.

Do **not** output any other text or free-form feedback. Do **not** give any judgement on the actual candidate, focus only on the the junior hiring manager's reason.

---  
Reference hiring reason provided by colleague senior hiring manager:  
{p1_reasoning}

Hiring reason provided by junior hiring manager:  
{p2_reasoning}"""

# Prompt for generating improved reasons
improve_reason_prompt = """You are a Senior Hiring Manager at Weights & Biases. You are tasked with creating a comprehensive, well-reasoned justification for a hiring decision (whether to interview a candidate or not).

Your reason should include a detailed analysis of the candidate in these THREE key areas:

1. Position Fit Analysis - Analyze how well the candidate's location, salary expectations, and general profile match the specific requirements in the job description. Include concrete details from both the application and job offer.

2. Experience Analysis - Evaluate whether the candidate's specific experiences, skills, and qualifications meet the job requirements. Compare their background directly to what the position needs.

3. Values Alignment - Assess how the candidate's demonstrated behaviors, accomplishments, and stated goals align with Weights & Biases values (Honesty, Curiosity, Gumption, Grit). Look for specific evidence in their application.

Your analysis of these three areas should adhere to these TWO important guidelines:

4. Evidence-Based Reasoning - Every claim you make must be supported by specific examples from the application or job description. Don't make general statements without backing them up with evidence.

5. Fairness & Objectivity - Ensure your evaluation uses consistent criteria across all aspects and avoids irrelevant factors or biases. Focus only on job-relevant qualifications and alignment.

Your reason should be comprehensive, well-structured, and exemplary. It should demonstrate clear critical thinking about the match between this specific candidate and this specific job.

Job Offer:
{offer_text}

Application:
{application_text}

Interview Decision: {interview_decision}

Write a detailed, well-reasoned justification for this decision that would score highly (4-5) on all analysis areas:"""

def download_dataset(artifact_path: str) -> Tuple[List[Dict[str, Any]], Any]:
    """
    Download dataset from W&B Artifacts using use_artifact to maintain lineage
    
    Args:
        artifact_path: Path to the artifact (e.g., 'wandb-smle/hiring-agent-demo-public/finetuning_dataset:v1')
        
    Returns:
        Tuple containing:
        - List of dataset examples (rows)
        - The initialized wandb run object
    """
    print(f"Downloading artifact: {artifact_path}")
    
    # Parse the artifact path
    parts = artifact_path.split("/")
    entity = parts[0]
    project = parts[1]
    artifact_name_version = parts[2]
    
    # Initialize a new wandb run for downloading
    run = wandb.init(project=project, entity=entity, job_type="download_dataset")
    
    # Use the artifact (this maintains lineage)
    artifact = run.use_artifact(artifact_path)
    
    # Get the artifact name which is also the table name
    artifact_name = artifact_name_version.split(':')[0]
    
    # Get the table directly using the artifact name as the table name
    try:
        table = artifact.get(artifact_name)
        
        # Convert table to a list of dictionaries
        rows = []
        for i in range(len(table.data)):
            row = {}
            for j, column in enumerate(table.columns):
                row[column] = table.data[i][j]
            rows.append(row)
        
        return rows, run
    except Exception as e:
        print(f"Error getting table: {e}")
        # If table retrieval fails, try to get all entries
        entries = artifact.get_entries()
        if entries:
            # Get the first available table
            table_name = list(entries.keys())[0]
            table = artifact.get(table_name)
            
            # Convert table to a list of dictionaries
            rows = []
            for i in range(len(table.data)):
                row = {}
                for j, column in enumerate(table.columns):
                    row[column] = table.data[i][j]
                rows.append(row)
            
            return rows, run
    
    # If we reach here, we couldn't get the data
    wandb.finish()
    raise ValueError(f"Could not extract dataset from artifact: {artifact_path}")

def improve_reason(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Improve the reason for the hiring decision using OpenAI
    
    Args:
        example: Dataset example containing offer_text, application_text, and interview
        
    Returns:
        Updated example with improved reason
    """
    offer_text = example.get("offer_text", "")
    application_text = example.get("application_text", "")
    interview_decision = example.get("interview", False)
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Using a powerful model for high-quality reasons
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates detailed, well-reasoned hiring decisions."},
                {"role": "user", "content": improve_reason_prompt.format(
                    offer_text=offer_text[:8000],  # Truncate to avoid token limits
                    application_text=application_text[:8000],  # Truncate to avoid token limits
                    interview_decision="Proceed with interview" if interview_decision else "Do not proceed with interview"
                )}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Create a new example with the improved reason
        updated_example = example.copy()
        updated_example["reason"] = response.choices[0].message.content.strip()
        return updated_example
    except Exception as e:
        print(f"Error while improving reason: {e}")
        # Return the original example with a placeholder reason
        updated_example = example.copy()
        updated_example["reason"] = "Error generating improved reason."
        return updated_example

def process_dataset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process the dataset rows to improve reasons using parallel processing
    
    Args:
        rows: List of dataset example dictionaries
        
    Returns:
        List of processed examples
    """
    print(f"Processing {len(rows)} examples using {NUM_THREADS} parallel threads...")
    
    # Create a thread pool executor with specified number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all tasks and map them to the original examples
        future_to_example = {executor.submit(improve_reason, example): example for example in rows}
        
        # Create a list to store the results
        processed_rows = []
        
        # Use tqdm to track progress
        for future in tqdm(concurrent.futures.as_completed(future_to_example), total=len(rows)):
            try:
                # Get the result (improved example)
                processed_example = future.result()
                processed_rows.append(processed_example)
            except Exception as e:
                # If anything went wrong, log the error and add the original example
                print(f"Error processing example: {e}")
                original_example = future_to_example[future]
                processed_rows.append(original_example)
    
    return processed_rows

def upload_dataset(rows: List[Dict[str, Any]], artifact_path: str, run: Any) -> str:
    """
    Upload the processed dataset to W&B and Weave
    
    Args:
        rows: List of processed dataset example dictionaries
        artifact_path: The original artifact path
        run: The active wandb run
        
    Returns:
        The new artifact path
    """
    # Parse the artifact path
    parts = artifact_path.split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid artifact path: {artifact_path}")
    
    entity = parts[0]
    project = parts[1]
    name_version = parts[2]
    
    # Split name and version
    name_parts = name_version.split(":")
    name = name_parts[0]
    
    # Create a new artifact
    artifact = wandb.Artifact(
        name=name,
        type="dataset",
        description=f"Improved reasons for {name}"
    )
    
    # Extract column names from the first row
    if rows:
        columns = list(rows[0].keys())
        
        # Create a wandb.Table with the rows
        table = wandb.Table(columns=columns)
        for row in rows:
            # Ensure all rows have all columns
            row_values = [row.get(col, None) for col in columns]
            table.add_data(*row_values)
        
        # Add the table to the artifact with the same name as the artifact
        artifact.add(table, name)
    
    # Log the artifact with the annotated alias
    run.log_artifact(artifact, aliases=["annotated"])
    
    # Save to Weave as well - Initialize weave first
    weave.init(f"{entity}/{project}")
    
    weave_dataset = weave.Dataset(name=name, rows=rows)
    weave_ref = weave.publish(weave_dataset)
    weave_url = urls.object_version_path(
        weave_ref.entity,
        weave_ref.project,
        weave_ref.name,
        weave_ref.digest,
    )
    
    print(f"Weave dataset published at: {weave_url}")
    
    # Return the new artifact path
    return f"{entity}/{project}/{name}:annotated"

def main():
    # Define the artifact paths
    artifact_paths = [
        "wandb-smle/hiring-agent-demo-public/finetuning_dataset:v1",
        "wandb-smle/hiring-agent-demo-public/evaluation_dataset:v1"
    ]
    
    # Process each artifact
    for artifact_path in artifact_paths:
        print(f"\nProcessing artifact: {artifact_path}")
        
        # Download the dataset - get rows and active run
        rows, run = download_dataset(artifact_path)
        
        # Process the dataset
        processed_rows = process_dataset(rows)
        
        # Upload the processed dataset (using the same run)
        new_artifact_path = upload_dataset(processed_rows, artifact_path, run)
        
        # Finish the run
        wandb.finish()
        
        print(f"Uploaded processed dataset to: {new_artifact_path}")

if __name__ == "__main__":
    main() 