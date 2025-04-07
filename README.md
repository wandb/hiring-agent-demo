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
Once the operator UI is launched through streamlit there are four modes that can be select through the dropdown menu under "Select Mode". If you create a new project make sure to **execute them chronologically**: 
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
pip install -r requirements.txt
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
2. Paste in artifact path into config panel and "Add Model to Ollama"
    - This will download the artifact from wandb
    - Will then call `ollama create fine-tuned-comparison-model -f Modelfile` from the root of the downloaded artifact (where the fine-tuning notebook adds a Modelfile automatically)
3. Now when you paste in the artifact link into the textfield it will always call the model with Ollama
    - To use one of the other models remove all text from the textfield

## Set fine-tuned model as endpoint for Weave playground
1. Make sure the model runs on Ollama (following above guide will suffice)
2. Start ngrok: `ngrok http 11434 --response-header-add "Access-Control-Allow-Origin: *" --host-header rewrite`
3. Configure in Weave UI
    - Add ngrok address without the `v1`
    - Add random secret
4. Open Playground and debug with the actual model
