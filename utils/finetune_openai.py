import os
import json
import wandb
import weave
import numpy as np
import tiktoken
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
from wandb.integration.openai.fine_tuning import WandbLogger
from dotenv import load_dotenv

load_dotenv(".env")

# Initialize W&B run
run = wandb.init(project=os.environ.get("WANDB_PROJECT", "openai-fine-tuning"))
entity = os.environ.get("WANDB_ENTITY", "wandb-smle")
project = os.environ.get("WANDB_PROJECT", "hiring-agent-demo-public")

# Function to validate dataset format and analyze token distribution
def validate_dataset(dataset_path, verbose=True):
    # Load dataset
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f]
    
    if verbose:
        print(f"Num examples in {dataset_path}: {len(dataset)}")
        print("First example:")
        for message in dataset[0]["messages"]:
            print(message)
    
    # Format error checks
    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
                
            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1
                
            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1
                
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1
    
    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
        raise ValueError("Format errors found in dataset")
    elif verbose:
        print("No format errors found")
    
    # Token counting functions
    encoding = tiktoken.get_encoding("cl100k_base")
    
    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    
    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens
    
    # Analyze token distribution
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []
    
    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
    
    if verbose:
        print("Num examples missing system message:", n_missing_system)
        print("Num examples missing user message:", n_missing_user)
        print("\nDistribution of num_messages_per_example:")
        print(f"min / max: {min(n_messages)}, {max(n_messages)}")
        print(f"mean / median: {np.mean(n_messages)}, {np.median(n_messages)}")
        
        print("\nDistribution of num_total_tokens_per_example:")
        print(f"min / max: {min(convo_lens)}, {max(convo_lens)}")
        print(f"mean / median: {np.mean(convo_lens)}, {np.median(convo_lens)}")
        
        print("\nDistribution of num_assistant_tokens_per_example:")
        print(f"min / max: {min(assistant_message_lens)}, {max(assistant_message_lens)}")
        print(f"mean / median: {np.mean(assistant_message_lens)}, {np.median(assistant_message_lens)}")
        
    n_too_long = sum(l > 4096 for l in convo_lens)
    if n_too_long > 0 and verbose:
        print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")
    
    # Calculate pricing and epoch estimate
    MAX_TOKENS_PER_EXAMPLE = 4096
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    TARGET_EPOCHS = 3
    MIN_EPOCHS = 1
    MAX_EPOCHS = 25
    
    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)
    
    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    
    if verbose:
        print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
        print(f"By default, you'll train for {n_epochs} epochs on this dataset")
        print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
    
    result = {
        "num_examples": len(dataset),
        "avg_tokens_per_example": np.mean(convo_lens),
        "n_too_long": n_too_long,
        "recommended_epochs": n_epochs,
        "total_tokens": n_billing_tokens_in_dataset
    }
    
    return result

# Function to format data for fine-tuning
def format_data(df, prompt_template):
    formatted_data = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        messages = []

        # Add system message
        system_message = "You are an AI assistant that evaluates job applications against job offers."
        messages.append({"role": "system", "content": system_message})
        
        # Format user prompt
        prompt = prompt_template.format(
            job_offer_extract=row["offer_text"],
            application_extract=row["application_text"]
        )
        messages.append({"role": "user", "content": prompt})
        
        # Format assistant response
        response = f"Interview: {'yes' if row['interview'] else 'no'}, Reason: {row['reason']}"
        messages.append({"role": "assistant", "content": response})

        formatted_data.append({"messages": messages})
    return formatted_data

# Function to retry on OpenAI API errors
@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def create_fine_tuning_job(openai_client, training_file_id, validation_file_id=None, model="gpt-3.5-turbo", n_epochs=3):
    """Create a fine-tuning job with retry logic"""
    params = {
        "training_file": training_file_id,
        "model": model,
        "hyperparameters": {
            "n_epochs": n_epochs
        }
    }
    
    if validation_file_id:
        params["validation_file"] = validation_file_id
    
    return openai_client.fine_tuning.jobs.create(**params)

# Get training dataset from WandB
print("Retrieving training dataset from Weights & Biases...")
training_artifact = run.use_artifact(f"{entity}/{project}/finetuning_dataset:latest", type='dataset')
training_dir = training_artifact.download()
training_table = training_artifact.get("finetuning_dataset")
training_df = training_table.get_dataframe()
print(f"Retrieved training dataset with {len(training_df)} examples")

# Get validation dataset from WandB
print("Retrieving validation dataset from Weights & Biases...")
validation_artifact = run.use_artifact(f"{entity}/{project}/evaluation_dataset:latest", type='dataset')
validation_dir = validation_artifact.download()
validation_table = validation_artifact.get("evaluation_dataset")
validation_df = validation_table.get_dataframe()
print(f"Retrieved validation dataset with {len(validation_df)} examples")

# Get prompt for conversation formatting
print("Getting prompt template...")
client = weave.init(f"{entity}/{project}")
compare_offer_application_prompt = weave.ref(f"weave:///{entity}/{project}/object/compare_offer_application_prompt:latest").get()

# Format data for OpenAI fine-tuning
print("Formatting training data for OpenAI fine-tuning...")
train_data = format_data(training_df, compare_offer_application_prompt)

print("Formatting validation data for OpenAI fine-tuning...")
valid_data = format_data(validation_df, compare_offer_application_prompt)

# Save formatted data to jsonl files
training_file_path = os.path.join(training_dir, "training_data.jsonl")
validation_file_path = os.path.join(validation_dir, "validation_data.jsonl")

with open(training_file_path, "w") as f:
    for entry in train_data:
        f.write(json.dumps(entry) + "\n")

with open(validation_file_path, "w") as f:
    for entry in valid_data:
        f.write(json.dumps(entry) + "\n")

# Validate the training and validation data
print("Validating training data...")
train_stats = validate_dataset(training_file_path)
print("\nValidating validation data...")
valid_stats = validate_dataset(validation_file_path)

# Log dataset stats to wandb
wandb.log({
    "train/num_examples": train_stats["num_examples"],
    "train/avg_tokens_per_example": train_stats["avg_tokens_per_example"],
    "train/total_tokens": train_stats["total_tokens"],
    "valid/num_examples": valid_stats["num_examples"],
    "valid/avg_tokens_per_example": valid_stats["avg_tokens_per_example"],
    "valid/total_tokens": valid_stats["total_tokens"]
})

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Upload the training file to OpenAI
print("Uploading training data to OpenAI...")
with open(training_file_path, "rb") as file:
    training_file = openai_client.files.create(
        file=file,
        purpose="fine-tune"
    )

# Upload the validation file to OpenAI
print("Uploading validation data to OpenAI...")
with open(validation_file_path, "rb") as file:
    validation_file = openai_client.files.create(
        file=file,
        purpose="fine-tune"
    )

# Create a fine-tuning job with the recommended number of epochs
print("Creating fine-tuning job...")
n_epochs = max(train_stats["recommended_epochs"], valid_stats["recommended_epochs"])
fine_tuning_job = create_fine_tuning_job(
    openai_client=openai_client,
    training_file_id=training_file.id,
    validation_file_id=validation_file.id,
    model='gpt-4o-mini-2024-07-18',  # Try specific dated version
    n_epochs=n_epochs
)

print(f"Fine-tuning job created: {fine_tuning_job.id}")
wandb.log({"fine_tune_job_id": fine_tuning_job.id})

# Sync fine-tuning results with W&B
print("Syncing fine-tuning results with Weights & Biases...")
WandbLogger.sync(
    fine_tune_job_id=fine_tuning_job.id,
    openai_client=openai_client,
    project=project,
    entity=entity,
    wait_for_job_success=False,
    model_artifact_name="gpt4o_mini_model"
)

print("Fine-tuning job synced with W&B successfully")

# Optional: Evaluation after fine-tuning
if os.environ.get("RUN_EVALUATION", "false").lower() == "true":
    print("Running evaluation on fine-tuned model...")
    
    # Wait for the fine-tuning job to complete
    print("Waiting for fine-tuning job to complete...")
    job_status = "pending"
    while job_status not in ["succeeded", "failed"]:
        job = openai_client.fine_tuning.jobs.retrieve(fine_tuning_job.id)
        job_status = job.status
        if job_status == "succeeded":
            print(f"Fine-tuning job completed successfully")
            break
        elif job_status == "failed":
            print(f"Fine-tuning job failed")
            run.finish()
            exit(1)
        else:
            print(f"Fine-tuning job status: {job_status}")
            import time
            time.sleep(60)  # Wait for 60 seconds before checking again
    
    # Get the fine-tuned model ID directly from the job
    fine_tuned_model_id = job.fine_tuned_model
    print(f"Using fine-tuned model: {fine_tuned_model_id}")
    
    # Create prediction table
    prediction_table = wandb.Table(columns=["input", "predicted", "target", "is_correct"])
    
    # Evaluate on validation data
    correct = 0
    total = 0
    
    for entry in tqdm(valid_data):
        messages = entry["messages"][:-1]  # Remove the assistant message
        target = entry["messages"][-1]["content"]
        
        # Get prediction from fine-tuned model
        response = openai_client.chat.completions.create(
            model=fine_tuned_model_id,
            messages=messages,
            max_tokens=100
        )
        
        prediction = response.choices[0].message.content
        is_correct = prediction.lower() == target.lower()
        
        if is_correct:
            correct += 1
        total += 1
        
        prediction_table.add_data(
            messages[-1]["content"],  # Input (user message)
            prediction,               # Predicted output
            target,                   # Target output
            is_correct                # Is the prediction correct
        )
    
    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation accuracy: {accuracy:.2f}")
    
    # Log evaluation results
    wandb.log({
        "eval/accuracy": accuracy,
        "eval/predictions": prediction_table
    })

print("Fine-tuning process completed successfully")
run.finish() 