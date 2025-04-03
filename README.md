# Hiring Agent Demo
E2E Models + Weave demo. Also serves as the demo project for the EU AI Act. 

## Setup
1. Create `utils/.env` and install `requirements.txt`
2. Run `python runapp.py` from root
3. Generate dataset and create base prompts
    - From config panel select "Create Dataset" and go through all steps
    - From config panel select "Manage Prompts" and publish all defaults for every tab
4. Run single test or whole evaluation

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
 