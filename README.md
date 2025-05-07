# Hiring Agent Demo
E2E Models + Weave demo. Also serves as the demo project for the EU AI Act. 

This repository contains a demo of a hiring agent that can evaluate job applications against job offers using Langgraph and multiple LLM providers.

## Features
- Automated evaluation of job applications against requirements
- Support for multiple LLM providers (OpenAI, AWS Bedrock, Ollama)
- PDF processing capabilities (multi-modal visualization)
- Hallucination detection and guardrails (incl. self-relfection and HITL)
- Expert review system
- Integration with Weights & Biases Models and Weave

## Usage
Once the operator UI is launched through streamlit there are four modes that can be select through the dropdown menu under "Select Mode". It is important to note currently the current pipeline expects local PDFs to make it more realistic, that means that the evaluation dataset contains paths to local PDF files. If you want to run an evaluation locally make sure to generate your own dataset first, specifically **execute the following steps chronologically chronologically**. : 
1. `Create Dataset`
    - Drag in job positions as PDFs (e.g. downloading wandb job positions)
    - Generate applicant characteristics table
    - Go to next step and calculate R score (no changes needed)
    - Go to next step and generate actual evaluation and fine-tuning dataset
2. `Manage Prompts`
    - If it's the first time you're running the project click `Publish Context Prompt` for every tab (change prompt if you want)
3. `Single Test`
    - Drag in one of the job position PDFs and one of the generated application PDFs (under `utils/data/applications`)
    - Decide whether to use a hallucination guardrail on the hiring reason in the config panel on the left
    - Decide whether to enable expert reviews (by default means if guardrail fails twice even after self-reflecting the first time)
4. `Batch Testing`
    - Turn expert review mode off (not compatible because of parallel evaluation yet)
    - Paste in weave URL of evaluation dataset that you generated
    - Run evaluation

To run the fine-tuned comparison model first click on "Add Model to Ollama" if you haven't yet installed the model locally and then select `custom-wandb-artifact-model` in the "Comparison Model" dropdown.

## Setup
1. (Recommended) Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt # requirements_relaxed.txt if problems
```
2. Create `utils/.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key  # If using AWS Bedrock
AWS_SECRET_ACCESS_KEY=your_aws_secret_key  # If using AWS Bedrock
AWS_DEFAULT_REGION=your_aws_region  
WANDB_API_KEY=your_wandb_api_key 
```
3. Run `python runapp.py` from root
4. Generate dataset and create base prompts
    - From config panel select "Create Dataset" and go through all steps
    - From config panel select "Manage Prompts" and publish all defaults for every tab
5. Run single test or whole evaluation

## Use fine-tuned comparison model
1. Based on your dataset fine-tune your comparison model in [this notebook](https://colab.research.google.com/drive/1zfhbL9KwUbbCcSvy46alJDCZY7TwSVIO?usp=sharing)
2. Paste in artifact path into config panel, select `custom-wandb-artifact-model` under "Comparison Model" and click the button "Add Model to Ollama"
    - This will download the artifact from wandb
    - Will then call `ollama create <model-name> -f Modelfile` from the root of the downloaded artifact (where the fine-tuning notebook adds a Modelfile automatically)
3. If you want to use parallel calls make sure to serve the ollama server with `OLLAMA_NUM_PARALLEL=<number-of-parallel-calls> ollama serve`
4. Now when use the single evaluation or the batch testing mode with the model

## Improve reason labels in datasets
The repository includes a script for enhancing the reason labels in training and evaluation datasets:
```bash
python improve_reason_labels.py
```

This script:
- Downloads the finetuning and evaluation datasets from W&B artifacts
- Uses OpenAI's GPT-4o to generate high-quality, detailed hiring reasons
- Reasons are structured to analyze position fit, experience, and values alignment
- Processes examples in parallel (10 threads) for faster execution
- Maintains proper artifact lineage with W&B
- Publishes improved datasets to both W&B (with "annotated" alias) and Weave

Run this script to generate better ground truth reasons for evaluating hiring agent performance.

## GPT-4o-mini Fine-Tuning with Weights & Biases

This repository includes a script for fine-tuning OpenAI's GPT-4o-mini model using datasets stored in W&B:

```bash
python utils/fine_tune_gpt4o_mini.py
```

### Features
- Retrieves dataset from W&B artifacts
- Validates and analyzes the dataset (token counts, format checking)
- Splits data into training and validation sets
- Recommends optimal epochs based on dataset size
- Provides token usage estimates for cost planning
- Logs comprehensive metrics to W&B dashboard
- Optional evaluation on the fine-tuned model

### Prerequisites
- OpenAI API key with fine-tuning access
- W&B account and properly formatted dataset

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export WANDB_API_KEY="your_wandb_api_key" 
   export WANDB_ENTITY="your_wandb_username_or_team"
   export WANDB_PROJECT="your_wandb_project"
   # Optional: Enable evaluation after fine-tuning
   export RUN_EVALUATION="true"
   ```

The fine-tuning progress can be monitored in your W&B dashboard, with logs of training metrics, dataset statistics, and model performance.

## Set fine-tuned model as endpoint for Weave playground
1. Make sure the model runs on Ollama (following above guide will suffice)
2. Start ngrok: `ngrok http 11434 --response-header-add "Access-Control-Allow-Origin: *" --host-header rewrite`
3. Configure in Weave UI
    - Add ngrok address without the `v1`
    - Add random secret
4. Open Playground and debug with the actual model
