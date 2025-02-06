# server.py
import weave
from weave.scorers import HallucinationFreeScorer
from weave.scorers.hallucination_scorer import HallucinationResponse
from weave.trace.api import get_current_call
from weave.trace import urls
from weave.integrations.bedrock.bedrock_sdk import patch_client
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langgraph.graph import END, StateGraph, START
from openai import OpenAI

import boto3
import os, sys
import asyncio
from PIL import Image
from typing import Optional, List, Any, Union
from typing_extensions import TypedDict
from pydantic import PrivateAttr
import streamlit as st

from utils.prompt import (
    CV, Offer, InterviewDecision, 
    context_prompt, comparison_prompt, guardrail_prompt,
    extract_offer_prompt, extract_application_prompt, compare_offer_application_prompt)
from utils.prepro import extract_text_from_pdf, pdf_to_images
from utils.evaluate import generate_dataset, decision_match, ReasonScorer
from dotenv import load_dotenv

# TODO: change last_comparison to take in a weave call object
# TODO: check difference between pydantic structured outcome defintion and custom prompts
# TODO: check whether better to define the nodes as classes with args (instead of partials) 
# Define the data model for the graph state
class GraphState(TypedDict):
    offer_pdf: str
    job_offer_extract: str
    application_pdf: str
    application_extract: str
    reason : str
    has_hallucination: bool
    decision: bool
    tries: int
    last_comparison: Any

class ExtractJobOffer:
    def __init__(self, extracttion_model) -> None:
        self.extraction_model = extraction_model

    @weave.op(name="ExtractJobOfferCall")
    def __call__(self, state: GraphState):
        """Extract the information from the job offer"""
        job_offer = extract_text_from_pdf(state["offer_pdf"])
        model = ChatOpenAI(model=self.extraction_model)
        job_offer_extract = model.invoke(extract_offer_prompt.format(job_offer=job_offer))
        state["job_offer_extract"] = job_offer_extract
        return state
    
class ExtractApplication:
    def __init__(self, extraction_model) -> None:
        self.extraction_model = extraction_model

    @weave.op(name="ExtractApplicationCall")
    def __call__(self, state: GraphState):
        """Extract the information from the application"""
        application = extract_text_from_pdf(state["application_pdf"])
        model = ChatOpenAI(model=self.extraction_model)
        application_extract = model.invoke(extract_application_prompt.format(application=application))
        state["application_extract"] = application_extract
        return state

class CompareApplicationOffer:
    def __init__(self, comparison_model) -> None:
        self.comparison_model = comparison_model

    @weave.op(name="CompareApplicationOfferCall")
    def __call__(self, state: GraphState):
        """Compare the application and offer and decide if they are fitting and why."""
        state["tries"] = 1 if not state.get("tries") else state.get("tries")+1
        application_extract = state["application_extract"]
        job_offer_extract = state["job_offer_extract"]
        # model = ChatBedrock(
        #     model_id=self.comparison_model,
        #     model_kwargs=dict(temperature=0),
        #     region_name="us-west-2",
        #     aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"], 
        #     aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"], 
        #     aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        # ).with_structured_output(InterviewDecision)
        model = ChatOpenAI(
            model=self.comparison_model,
            response_format={"type": "json"}).with_structured_output(InterviewDecision)
        comparison_document = compare_offer_application_prompt.format(
            job_offer_extract=job_offer_extract, 
            application_extract=application_extract
        )
        decision = model.invoke(comparison_document)
        state["reason"] = decision.reason
        state["decision"] = decision.decision
        return state

class HallucinationGuardrail:
    def __init__(self, guardrail_model) -> None:
        self.guardrail_model = guardrail_model

    @weave.op(name="HallucinationGuardrail")
    def __call__(self, state: GraphState):
        """Use guardrail to check whether reason only contains info from application or offer"""
        reason = state["reason"]
        application_extract = state["application_extract"]
        job_offer_extract = state["job_offer_extract"]
        hallucination_scorer = HallucinationFreeScorer(
            client=OpenAI(),
            model_id=self.guardrail_model,
        )
        context_document = context_prompt.format(
            job_offer_extract=job_offer_extract,
            application_extract=application_extract
        )
        guardrail_result = hallucination_scorer.score(output=reason, context=context_document)
        state["has_hallucination"] = guardrail_result["has_hallucination"]
        return state

@weave.op
def extract_application(extraction_model: str, state: GraphState):
    """Extract the information from the application """
    application = extract_text_from_pdf(state["application_pdf"])
    model = ChatOpenAI(model=extraction_model)
    application_extract = model.invoke(extract_application_prompt.format(application=application))
    state["application_extract"] = application_extract
    return state


@weave.op
def compare_application_offer(comparison_model: str, state: GraphState):
    """Compare the application and offer and decide if they are fitting and why. """
    application_extract = state["application_extract"]
    job_offer_extract = state["job_offer_extract"]
    model = ChatOpenAI(
        model=comparison_model,
        response_format={"type": "json"}).with_structured_output(InterviewDecision)
    comparison_document = compare_offer_application_prompt.format(job_offer_extract=job_offer_extract, application_extract=application_extract)
    decision = model.invoke(comparison_document)
    state["reason"] = decision.reason
    state["decision"] = decision.decision
    state["last_comparison"] = weave.get_current_call()
    return state

@weave.op
def hallucination_guardrail(guardrail_model: str, state: GraphState):
    """Use guardrail to check whether reason only contains info from application or offer"""
    reason = state["reason"]
    application_extract = state["application_extract"]
    job_offer_extract = state["job_offer_extract"]
    hallucination_scorer = HallucinationFreeScorer(
        client=OpenAI(),
        model_id=guardrail_model,
    )
    context_document = context_prompt.format(job_offer_extract=job_offer_extract,application_extract=application_extract)
    guardrail_result = hallucination_scorer.score(output=reason, context=context_document)

    # TODO: I could define a column map in the scorer to map the decision.reason to the "output" (not sure because it's only meant for dataset right)?
    #       Still how do I get in the context_documents? I would need to retrieve them from the beginning of the whole trace? 
    #       We need some mapping for the call object? Also what's the benefit of applying the score instead of calling .score? (the extra field in the UI?)
    # guardrail_result = state["last_comparison"].apply_scorer(hallucination_scorer)

    state["has_hallucination"] = guardrail_result["has_hallucination"]
    return state

@weave.op
def expert_review(state: GraphState):
    """Mark call as needing expert review using the "hitl" tag and wait for expert to provide decision and reason."""
    # Display expert review panel in streamlit
    st.header("Expert Review Required")
    
    st.markdown("### Decision Context")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Job Offer Details")
        st.write(state["job_offer_extract"])
        
    with col2:
        st.subheader("Application Details") 
        st.write(state["application_extract"])
    
    st.markdown("### Current Model Output")
    st.subheader("Decision")
    st.write(state["decision"])
    st.subheader("Reasoning")
    st.write(state["reason"])
    
    # Use form to prevent refresh on input
    with st.form("expert_review_form"):
        st.markdown("### Expert Override")
        corrected_reason = st.text_area("Corrected Reasoning", value=state["reason"])
        corrected_decision = st.selectbox("Final Decision", ["INTERVIEW", "NO_INTERVIEW"])
        
        submitted = st.form_submit_button("Submit Expert Decision")
        if submitted:
            state["decision"] = corrected_decision == "INTERVIEW"
            state["reason"] = corrected_reason
            st.success("Expert decision recorded successfully!")
            
    return state


@weave.op
def validate(state: GraphState) -> str:
    """
    Decides whether to retry based on the feedback from the guardrail and the number of tries.

    Args:
        state (dict): The current graph state

    Returns:
        str: decision for next node to call
    """

    print("--- Decide to validate or retry ---")

    has_hallucination = state["has_hallucination"]
    if has_hallucination == True and state["tries"] < 2:
        print("---DECISION: Hallucination detected, will retry---")
        return "retry"
    elif has_hallucination == True and state["tries"] >= 2:
        print("---DECISION: Hallucination detected, too many retries, reaching to expert---")
        return "hitl"
    else:
        print("---DECISION: Answer Accepted---")
        return "valid"

def create_wf(
        extraction_model: str,
        comparison_model: str,
        guardrail_model: str):
    # Define the nodes in the workflow
    # Initialize the workflow
    workflow = StateGraph(GraphState)
    workflow.add_node("extract_job_offer", ExtractJobOffer(extracttion_model=extraction_model))
    workflow.add_node("extract_application", ExtractApplication(extraction_model=extraction_model))
    workflow.add_node("compare_application_offer", CompareApplicationOffer(comparison_model=comparison_model))
    workflow.add_node("hallucination_guardrail", HallucinationGuardrail(guardrail_model=guardrail_model))
    workflow.add_node("expert_review", expert_review)

    # Start from 'generate_application' step
    workflow.add_edge(START, "extract_job_offer")
    workflow.add_edge("extract_job_offer", "extract_application")
    workflow.add_edge("extract_application", "compare_application_offer")
    workflow.add_edge("compare_application_offer", "hallucination_guardrail")
    workflow.add_conditional_edges(
        "hallucination_guardrail",
        validate,
        {
            "retry": "compare_application_offer",
            "hitl": "expert_review",
            "valid": END,
        },
    )
    workflow.add_edge("expert_review", END)

    # Compile the workflow
    app = workflow.compile()
    return app

# Function to visualize the workflow graph
def create_wf_graph(app):
    try:
        # Generate the graph as a PNG file
        graph_png = app.get_graph().draw_mermaid_png()

        # Save the PNG to a file
        output_path = "workflow_graph.png"
        with open(output_path, "wb") as f:
            f.write(graph_png)

        # Open the image using PIL (Python Imaging Library)
        img = Image.open(output_path)
        #img.show()  # This will open the image using the default image viewer

        print(f"Graph has been saved to {output_path} and displayed.")

    except Exception as e:
        # If any error occurs (for example, missing dependencies), print the error.
        print(f"An error occurred: {e}")

def stream_graph_updates(app, offer_pdf: str, application_pdf: str):
    # Stream the workflow updates
    for event in app.stream({"offer_pdf": offer_pdf, "application_pdf": application_pdf}):
        for value in event.values():
            print("Assistant:", value)
            reason = value.get("reason", "")
            decision = value.get("decision", "")
            has_hallucination = value.get("has_hallucination", "")
    return {
            'interview': decision,
            'reason': reason,
            'has_hallucination': has_hallucination
        }

class HiringAgent(weave.Model):
    """HiringAgent based on OpenAI with Guardrail."""
    extraction_model: str
    comparison_model: str
    guardrail_model: str
    _app : Any = PrivateAttr()

    # Start streaming the graph updates using job_offer and cv as inputs
    def model_post_init(self, __context):
        self._app = create_wf(extraction_model, comparison_model, guardrail_model) 

    @weave.op
    def predict(self, offer_pdf: str, application_pdf: str,
                offer_images: Optional[Union[List[Image.Image], Image.Image]] = None,
                application_images: Optional[Union[List[Image.Image], Image.Image]] = None) -> dict:
        # extraction
        return stream_graph_updates(self._app, offer_pdf, application_pdf)

if __name__ == "__main__":
    # Setup
    load_dotenv("utils/.env")
    #client = weave.init("hhuml/demo-hiring-agent")
    client = weave.init("wandb-smle/e2e-hiring-assistant")
    st.set_page_config(page_title="Hiring Assistant", layout="wide")

    st.title("Hiring Assistant")
    # Create and patch the Bedrock client
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
    )
    # patching step for Weave
    patch_client(client)

    # Sidebar for mode selection and model configuration
    with st.sidebar:
        st.header("Configuration")
        mode = st.selectbox("Select Mode", ["Single Test", "Batch Testing", "Create Dataset"])
        
        st.subheader("Model Settings")
        extraction_model = st.selectbox("Extraction Model", ["gpt-4o-mini", "gpt-4o"])
        comparison_model = st.selectbox("Comparison Model", ["anthropic.claude-3-sonnet-20240229-v1:0", "us.amazon.nova-lite-v1:0", "gpt-4o-mini", "gpt-4o"]) 
        guardrail_model = st.selectbox("Guardrail Model", ["gpt-4o-mini", "smollm2-135m-v18"])
        use_guardrail = st.checkbox("Use Guardrail", value=True)

        st.subheader("Evaluation Settings")
        judge_model = st.selectbox("Judge Model", ["gpt-4o-mini", "gpt-4o"])

    # Initialize agent with selected configuration
    hiring_agent = HiringAgent(
        extraction_model=extraction_model,
        comparison_model=comparison_model,
        guardrail_model=guardrail_model
    )

    if mode == "Single Test":
        st.header("Upload Documents")
        col1, col2 = st.columns(2)
        
        with col1:
            offer_file = st.file_uploader("Upload Job Offer PDF", type=['pdf'])
            if offer_file:
                with open("temp_offer.pdf", "wb") as f:
                    f.write(offer_file.getbuffer())
                st.success("Offer PDF uploaded successfully")
                
        with col2:
            application_file = st.file_uploader("Upload Application PDF", type=['pdf'])
            if application_file:
                with open("temp_application.pdf", "wb") as f:
                    f.write(application_file.getbuffer())
                st.success("Application PDF uploaded successfully")

        if offer_file and application_file:
            if st.button("Evaluate Application"):
                with st.spinner("Processing documents..."):
                    offer_images = pdf_to_images("temp_offer.pdf")
                    application_images = pdf_to_images("temp_application.pdf")
                    
                    result = hiring_agent.predict(
                        offer_pdf="temp_offer.pdf",
                        application_pdf="temp_application.pdf",
                        offer_images=offer_images,
                        application_images=application_images
                    )

                st.header("Results")
                if result['interview'] == 'PENDING_REVIEW':
                    st.warning("⚠️ Decision Pending Expert Review")
                elif result['interview']:
                    st.success("✅ Recommend to Interview")
                else:
                    st.error("❌ Do Not Recommend")
                
                st.subheader("Reasoning")
                st.write(result['reason'])

                # Cleanup
                os.remove("temp_offer.pdf")
                os.remove("temp_application.pdf")

    elif mode == "Batch Testing":
        st.header("Batch Evaluation")
        dataset_ref = st.text_input(
            "Dataset Reference", 
            "weave:///hhuml/demo-hiring-agent/object/evaluation_dataset:X4X7nNTPMllcxJ7JqW2KXJYggAXw7GMjoXQNzoqJwKs"
            #"weave:///wandb-smle/yyyy-hiring-assistant/object/evaluation_dataset:UOy8Zr7MYEOYJgoixA69tNi4ViuMKEjoTvYfTu1lz6U"
        )
        trials = st.number_input("Number of Trials", min_value=1, value=1)
        
        if st.button("Run Evaluation"):
            with st.spinner("Running batch evaluation..."):
                benchmark = weave.Evaluation(
                    dataset=weave.ref(dataset_ref).get(),
                    scorers=[decision_match, ReasonScorer(model_id=judge_model)],
                    preprocess_model_input=lambda row: {
                        "offer_pdf": row["offer_pdf"],
                        "offer_text": row["offer_text"],
                        "application_pdf": row["application_pdf"],
                        "application_text": row["application_text"],
                        "interview": row["interview"],
                        "reason": row["reason"],
                        "offer_images": pdf_to_images(str(row["offer_pdf"])),
                        "application_images": pdf_to_images(str(row["application_pdf"])),
                    },
                    trials=trials
                )
                results = asyncio.run(benchmark.evaluate(hiring_agent))
                st.json(results)

    else:  # Create Dataset
        st.header("Create Evaluation Dataset")
        num_applications = st.number_input("Number of Applications to Generate", min_value=1, value=1)
        offer_files = st.file_uploader("Upload Offer PDFs", type=['pdf'], accept_multiple_files=True)
        
        if offer_files and st.button("Generate Dataset"):
            with st.spinner("Generating dataset..."):
                # Create offers directory if it doesn't exist
                os.makedirs("./utils/data/offers", exist_ok=True)
                os.makedirs("./utils/data/applications", exist_ok=True)
                
                # Save uploaded files to local directory if they don't exist
                offer_paths = []
                for f in offer_files:
                    save_path = f"./utils/data/offers/{f.name}"
                    if not os.path.exists(save_path):
                        with open(save_path, "wb") as file:
                            file.write(f.getvalue())
                    offer_paths.append(save_path)
                
                dataset_struct = generate_dataset(
                    offer_list=offer_paths,
                    generation_model=extraction_model,
                    num_app=num_applications,
                )
                
                rows = [example.model_dump() for example in dataset_struct.examples]
                dataset = weave.Dataset(name="evaluation_dataset", rows=rows)
                ref = weave.publish(dataset)
                url = urls.object_version_path(
                    ref.entity,
                    ref.project,
                    ref.name,
                    ref.digest,
                )
                st.success(f"Dataset created and published successfully! [View dataset]({url})")


