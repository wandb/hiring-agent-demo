# server.py
import weave
from weave.scorers import HallucinationFreeScorer
from weave.scorers.hallucination_scorer import HallucinationResponse
from weave.trace.api import get_current_call
from weave.trace import urls
from weave.integrations.bedrock.bedrock_sdk import patch_client
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from openai import OpenAI

import boto3
import os, sys, json, subprocess, shutil
import asyncio, uuid
from PIL import Image
from typing import Optional, List, Any, Union
from typing_extensions import TypedDict
from pydantic import PrivateAttr
import streamlit as st

import pandas as pd
import numpy as np
import wandb
import time

from utils.prompt import (
    CV, Offer, InterviewDecision, 
    context_prompt, guardrail_prompt,
    extract_offer_prompt, extract_application_prompt, compare_offer_application_prompt)
from utils.prepro import extract_text_from_pdf, pdf_to_images, pre_process_eval
from utils.evaluate import decision_match, ReasonScorer
from utils.generate import generate_dataset, generate_applicant_characteristics, calculate_r_score, generate_application_from_characteristics
from dotenv import load_dotenv

# Initialize Streamlit session state variables
if 'thread_config' not in st.session_state:
    st.session_state.thread_config = {"configurable": {"thread_id": uuid.uuid4()}}
if 'interrupt' not in st.session_state:
    st.session_state.interrupt = False
if 'reason' not in st.session_state:
    st.session_state.reason = ""
if 'decision' not in st.session_state:
    st.session_state.decision = False
if 'has_hallucination' not in st.session_state:
    st.session_state.has_hallucination = False
if 'job_offer_extract' not in st.session_state:
    st.session_state.job_offer_extract = ""
if 'application_extract' not in st.session_state:
    st.session_state.application_extract = ""

# possible models
openai_models = ["gpt-4o-mini", "gpt-4o"]

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
    def __init__(self, extraction_model) -> None:
        self.extraction_model = extraction_model

    @weave.op(name="ExtractJobOfferCall")
    def __call__(self, state: GraphState):
        """Extract the information from the job offer"""
        job_offer = extract_text_from_pdf(state["offer_pdf"])
        model = ChatOpenAI(model=self.extraction_model)
        # Use wandb_entity and wandb_project from global scope
        latest_offer_prompt = weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/extract_offer_prompt:latest").get()
        job_offer_extract = model.invoke(latest_offer_prompt.format(job_offer=job_offer))
        state["job_offer_extract"] = job_offer_extract.content
        return state
    
class ExtractApplication:
    def __init__(self, extraction_model) -> None:
        self.extraction_model = extraction_model

    @weave.op(name="ExtractApplicationCall")
    def __call__(self, state: GraphState):
        """Extract the information from the application"""
        application = extract_text_from_pdf(state["application_pdf"])
        model = ChatOpenAI(model=self.extraction_model)
        # Use wandb_entity and wandb_project from global scope
        latest_application_prompt = weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/extract_application_prompt:latest").get()
        application_extract = model.invoke(latest_application_prompt.format(application=application))
        state["application_extract"] = application_extract.content
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
        
        if self.comparison_model in openai_models:
            model = ChatOpenAI(
                model=self.comparison_model,
                response_format={"type": "json"}).with_structured_output(InterviewDecision)
        elif self.comparison_model.startswith("wandb-artifact:///"):
            # NOTE: this doesn't work with weave.Evaluation and parallel processing
            # track inference 
            run = wandb.init(
                project=wandb_project, 
                entity=wandb_entity, 
                job_type="inference"
            )
            
            # Extract model name from the artifact path
            artifact_name = self.comparison_model.split("/")[-1]
            model_name = artifact_name.split(":")[0]

            # Use artifact without downloaded it for lineage
            artifact = run.use_artifact(artifact_name)

            # We can assume that model is already downloaded and added to Ollama
            model_name = "fine-tuned-comparison-model"
            model = ChatOllama(
                model=model_name,
                format="json"
            ).with_structured_output(InterviewDecision)
        else:
            model = ChatBedrock(
                model=self.comparison_model,
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                aws_session_token=os.environ["AWS_SESSION_TOKEN"]).with_structured_output(InterviewDecision)
            
        # Use wandb_entity and wandb_project from global scope
        latest_comparison_prompt = weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/compare_offer_application_prompt:latest").get()
        comparison_document = latest_comparison_prompt.format(
            job_offer_extract=job_offer_extract, 
            application_extract=application_extract
        )
        decision = model.invoke(comparison_document)
        state["reason"] = decision.reason
        state["decision"] = decision.decision
        
        if self.comparison_model.startswith("wandb-artifact:///"):
            run.log({
                "decision": decision.decision,
                "reason": decision.reason,
                "job_offer_extract": job_offer_extract,
                "application_extract": application_extract,
                "comparison_document": comparison_document
            })
            run.finish()
        return state

class HallucinationGuardrail:
    def __init__(self, guardrail_model) -> None:
        self.guardrail_model = guardrail_model

    @weave.op(name="HallucinationGuardrail")
    def __call__(self, state: GraphState):
        """Use guardrail to check whether reason only contains info from application or offer"""
        application_extract = state["application_extract"]
        job_offer_extract = state["job_offer_extract"]
        hallucination_scorer = HallucinationFreeScorer(
            model_id=f"openai/{self.guardrail_model}",
        )
        decision_reason = "Decision: We should move on with an interview\n" if state["decision"] else "Decision: We should NOT move on with an interview\n"
        decision_reason += f"Reason: {state['reason']}"
        latest_context_prompt = weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/context_prompt:latest").get()
        context_document = latest_context_prompt.format(
            job_offer_extract=job_offer_extract,
            application_extract=application_extract
        )

        # NOTE: we want this to run sync to know what next to do
        guardrail_result = asyncio.run(hallucination_scorer.score(output=decision_reason, context=context_document))

        # TODO: Check postprocess_output, does that exist for Scorers, Evaluations?
        # TODO: I could define a column map in the scorer to map the decision.reason to the "output" (not sure because it's only meant for dataset right)?
        #       Still how do I get in the context_documents? I would need to retrieve them from the beginning of the whole trace? 
        #       We need some mapping for the call object? Also what's the benefit of applying the score instead of calling .score? (the extra field in the UI?)
        # guardrail_result = state["last_comparison"].apply_scorer(hallucination_scorer)

        state["has_hallucination"] = guardrail_result["has_hallucination"]
        return state

@weave.op
def expert_review(state: GraphState):
    """Mark call as needing expert review using feedback instead of a tag."""
    # Mark the call as needing expert review using feedback
    current_call = get_current_call()
    if current_call:
        current_call.feedback.add("needs_expert_review", {"value": True})
    
    # Just pass through the state
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
        print("---DECISION: Hallucination detected, too many retries, marking for expert review---")
        return "expert_review"
    else:
        print("---DECISION: Answer Accepted---")
        return "valid"

def create_wf(
        extraction_model: str,
        comparison_model: str,
        guardrail_model: str,
        hitl_always_on: bool,
        disable_expert_review: bool = False):
    # Define the nodes in the workflow
    # Initialize the workflow
    workflow = StateGraph(GraphState)
    workflow.add_node("extract_job_offer", ExtractJobOffer(extraction_model=extraction_model))
    workflow.add_node("extract_application", ExtractApplication(extraction_model=extraction_model))
    workflow.add_node("compare_application_offer", CompareApplicationOffer(comparison_model=comparison_model))
    workflow.add_node("hallucination_guardrail", HallucinationGuardrail(guardrail_model=guardrail_model))
    workflow.add_node("expert_review", expert_review)

    # Start from 'generate_application' step
    workflow.add_edge(START, "extract_job_offer")
    workflow.add_edge("extract_job_offer", "extract_application")
    workflow.add_edge("extract_application", "compare_application_offer")
    workflow.add_edge("compare_application_offer", "hallucination_guardrail")
    
    # Configure the flow based on expert review settings
    if disable_expert_review:
        # Never use expert review, even for hallucinations
        workflow.add_edge("hallucination_guardrail", END)
    elif hitl_always_on:
        # Always use expert review
        workflow.add_edge("hallucination_guardrail", "expert_review")
        workflow.add_edge("expert_review", END)
    else:
        # Only use expert review when needed (default behavior)
        workflow.add_conditional_edges(
            "hallucination_guardrail",
            validate,
            {
                "retry": "compare_application_offer",
                "expert_review": "expert_review",
                "valid": END,
            },
        )
        workflow.add_edge("expert_review", END)

    # A checkpointer is required for LangGraph
    checkpointer = MemorySaver()

    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer)
    return app

# TODO: function to visualize the workflow graph
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

# Initialize key session state variables
if 'waiting_for_review' not in st.session_state:
    st.session_state.waiting_for_review = False
if 'call_id_for_review' not in st.session_state:
    st.session_state.call_id_for_review = None
if 'in_expert_review' not in st.session_state:
    st.session_state.in_expert_review = False
if 'expert_reason_input' not in st.session_state:
    st.session_state.expert_reason_input = ""
if 'expert_decision_input' not in st.session_state:
    st.session_state.expert_decision_input = "True"

# Initialize deprecated variables for backward compatibility
if 'expert_decision_spec_created' not in st.session_state:
    st.session_state.expert_decision_spec_created = False  # Deprecated but needed for compatibility
if 'expert_reason_spec_created' not in st.session_state:
    st.session_state.expert_reason_spec_created = False  # Deprecated but needed for compatibility

def submit_expert_review(call_id, decision, reason):
    """Submit expert review directly - no form needed."""
    print(f"submit_expert_review called with call_id={call_id}, decision={decision}")
    
    try:
        # Initialize Weave client
        client = weave.init(f"{st.session_state.wandb_entity}/{st.session_state.wandb_project}")
        
        # Get the call by ID
        call = client.get_call(call_id)
        
        # Convert decision to boolean for consistency
        decision_bool = decision == "True"
        
        # Add comprehensive feedback note for easy viewing
        call.feedback.add_note(f"Expert Review: {'Recommend' if decision_bool else 'Do Not Recommend'}")
        
        # Add structured feedback using the standard add method
        call.feedback.add("expert_review", {
            "decision": decision_bool,
            "reason": reason,
            "timestamp": time.time()
        })
        
        # Log to W&B for additional tracking
        run = wandb.init(
            project=st.session_state.wandb_project, 
            entity=st.session_state.wandb_entity, 
            job_type="expert_review"
        )
        run.log({
            "expert_decision": decision_bool,
            "expert_reason": reason,
            "call_id": call_id,
            "model_decision": st.session_state.decision,
            "model_reason": st.session_state.reason
        })
        run.finish()
        
        # Return success
        return True
        
    except Exception as e:
        print(f"Error in submit_expert_review: {str(e)}")
        st.error(f"Error adding feedback: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return False

def streamlit_expert_panel(): 
    """Display the expert review panel using direct button handling instead of a form."""
    st.header("Expert Review Required")
    
    # Set the in_expert_review flag to ensure we stay in this state during reloads
    st.session_state.in_expert_review = True
    
    # Get the call ID
    call_id = st.session_state.call_id_to_annotate
    st.info(f"Your review will be attached to call ID: {call_id}")
    
    # Show model output
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">', unsafe_allow_html=True)
        st.subheader("Job Offer Details")
        st.write(st.session_state.job_offer_extract)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div style="background-color: #fff0f5; padding: 10px; border-radius: 5px;">', unsafe_allow_html=True)
        st.subheader("Application Details") 
        st.write(st.session_state.application_extract)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Model Decision")
    st.write("Decision: ", "Recommend to Interview" if st.session_state.decision else "Do Not Recommend")
    st.write("Reasoning: ", st.session_state.reason)
    
    # Expert review input
    st.markdown("### Your Expert Review")
    
    # Initialize the expert reason with the model's reason if empty
    if not st.session_state.expert_reason_input:
        st.session_state.expert_reason_input = st.session_state.reason
        
    # Initialize the expert decision with the model's decision if not set
    if not st.session_state.expert_decision_input:
        st.session_state.expert_decision_input = "True" if st.session_state.decision else "False"
    
    # Create callback functions to update session state
    def update_reason():
        # This will be called when the text area changes
        print(f"Updating expert reason to: {st.session_state.expert_reason_temp}")
        st.session_state.expert_reason_input = st.session_state.expert_reason_temp
        
    def update_decision():
        # This will be called when the selectbox changes
        print(f"Updating expert decision to: {st.session_state.expert_decision_temp}")
        st.session_state.expert_decision_input = st.session_state.expert_decision_temp
    
    # Expert reasoning input with explicit key and on_change
    st.text_area(
        "Expert Reasoning", 
        value=st.session_state.expert_reason_input,
        key="expert_reason_temp",
        on_change=update_reason,
        height=200
    )
    
    # Expert decision input with explicit key and on_change
    st.selectbox(
        "Expert Decision", 
        options=["True", "False"],
        index=0 if st.session_state.expert_decision_input == "True" else 1,
        key="expert_decision_temp",
        on_change=update_decision
    )
    
    # Submit button - directly calls the function
    if st.button("Submit Expert Review", type="primary"):
        # Print debug values to confirm what we're using
        print("---- Expert Review Submission ----")
        print(f"Expert decision: {st.session_state.expert_decision_input}")
        print(f"Expert reason: {st.session_state.expert_reason_input}")
        
        # Verify we have the data we need
        if not call_id:
            st.error("Missing call ID")
            return
            
        if not st.session_state.expert_reason_input:
            st.error("Please provide reasoning")
            return
            
        # Process the submission
        with st.spinner("Submitting expert review..."):
            success = submit_expert_review(
                call_id=call_id,
                decision=st.session_state.expert_decision_input,
                reason=st.session_state.expert_reason_input
            )
            
        # Show result
        if success:
            st.success("🎉 Expert review successfully annotated!")
            st.session_state.waiting_for_review = False
            st.session_state.in_expert_review = False  # Clear the review flag on success
            
            # Show final decision
            st.header("Final Expert Decision")
            if st.session_state.expert_decision_input == "True":
                st.success("✅ Expert Recommends to Interview")
            else:
                st.error("❌ Expert Does Not Recommend")
            
            st.subheader("Expert Reasoning")
            st.write(st.session_state.expert_reason_input)
        else:
            st.error("Failed to submit expert review. Please try again.")

# Update stream_graph_updates to set waiting_for_review flag
def stream_graph_updates(app, offer_pdf: str, application_pdf: str):
    # Set up a thread ID for the graph
    if "thread_config" not in st.session_state:
        st.session_state.thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

    # Start a new event loop
    result = {}
    print("Starting HiringAgent prediction workflow...")
    
    try:
        # Execute the prediction
        for event in app.stream({"offer_pdf": offer_pdf, "application_pdf": application_pdf}, config=st.session_state.thread_config):
            for key, value in event.items():
                if key != "__interrupt__":  # Ignore any interrupt events
                    print(f"Event: {key}")
                    if isinstance(value, dict):
                        # Store results in session state
                        reason = value.get("reason", "")
                        decision = value.get("decision", "")
                        has_hallucination = value.get("has_hallucination", "")
                        job_offer_extract = value.get("job_offer_extract", "")
                        application_extract = value.get("application_extract", "")
                        
                        st.session_state.reason = reason
                        st.session_state.decision = decision
                        st.session_state.has_hallucination = has_hallucination
                        st.session_state.job_offer_extract = job_offer_extract
                        st.session_state.application_extract = application_extract
                        
                        # Update the result dictionary
                        result = {
                            'interview': decision,
                            'reason': reason,
                            'has_hallucination': has_hallucination
                        }
        
        # Get the current call ID and save it
        print("Prediction complete - Getting current call ID...")
        current_call = get_current_call()
        if current_call:
            call_id = current_call.id
            st.session_state.call_id_to_annotate = call_id
            print(f"Saved call ID for annotation: {call_id}")
        else:
            print("Warning: Could not get current call ID")
            # Fallback: try to get the most recent call
            try:
                client = weave.init(f"{st.session_state.wandb_entity}/{st.session_state.wandb_project}")
                calls = client.get_calls(
                    filter={"trace_roots_only": True},
                    sort_by=[{"field":"started_at","direction":"desc"}],
                    limit=1
                )
                if calls:
                    call_id = calls[0].id
                    st.session_state.call_id_to_annotate = call_id
                    print(f"Saved most recent call ID: {call_id}")
                else:
                    print("Warning: No recent calls found")
            except Exception as e:
                print(f"Error getting recent calls: {str(e)}")
        
        # Set expert review flag based on hallucination or settings
        if expert_review_options == "Always On" or has_hallucination:
            st.session_state.waiting_for_review = True
            print("Expert review needed")
        else:
            st.session_state.waiting_for_review = False
            print("No expert review needed")
        
    except Exception as e:
        print(f"Error in prediction workflow: {str(e)}")
        st.error(f"Error in prediction: {str(e)}")
    
    return result

class HiringAgent(weave.Model):
    """HiringAgent based on OpenAI with Guardrail."""
    extraction_model: str
    comparison_model: str
    guardrail_model: str
    hitl_always_on: bool = False
    disable_expert_review: bool = False
    _app : Any = PrivateAttr()

    # Start streaming the graph updates using job_offer and cv as inputs
    def model_post_init(self, __context):
        self._app = create_wf(
            self.extraction_model, 
            self.comparison_model, 
            self.guardrail_model, 
            self.hitl_always_on,
            self.disable_expert_review
        ) 

    @weave.op
    def predict(self, offer_pdf: str, application_pdf: str,
                offer_images: Optional[Union[List[Image.Image], Image.Image]] = None,
                application_images: Optional[Union[List[Image.Image], Image.Image]] = None) -> dict:
        # extraction
        result = stream_graph_updates(self._app, offer_pdf, application_pdf)
        return result
    
# Initialize session state for form handling
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'expert_decision' not in st.session_state:
    st.session_state.expert_decision = None
if 'expert_reason' not in st.session_state:
    st.session_state.expert_reason = None

# Debug info at the very beginning of the app
print(f"[APP START] form_submitted: {st.session_state.form_submitted}")
print(f"[APP START] expert_decision: {st.session_state.expert_decision}")
if st.session_state.form_submitted:
    print(f"[APP START] We should process expert review!")

# Function to save form data when submit button is pressed
def handle_form_submit():
    """This is called when the form submit button is clicked."""
    # Get values directly from the form fields
    decision = st.session_state.expert_decision
    reason = st.session_state.expert_reason
    
    # Log what we're doing
    print(f"Submit callback - setting form_submitted=True")
    print(f"Submit callback - decision: {decision}")
    print(f"Submit callback - reason length: {len(reason) if reason else 0}")
    
    # Set the form_submitted flag and session state values
    st.session_state.form_submitted = True
    
    # Just to be extra safe, set the values explicitly again
    st.session_state.expert_decision = decision
    st.session_state.expert_reason = reason

# Add the main app debug panel
def show_debug_panel():
    with st.expander("Debug Information", expanded=True):
        st.subheader("Session State")
        st.write("form_submitted:", st.session_state.form_submitted)
        st.write("expert_decision:", st.session_state.expert_decision)
        if st.session_state.expert_reason:
            st.write(f"expert_reason length: {len(st.session_state.expert_reason)}")
        st.write("call_id_to_annotate:", st.session_state.call_id_to_annotate)
        
        st.subheader("Debug Logs (most recent first)")
        if 'debug_logs' in st.session_state and st.session_state.debug_logs:
            logs = list(st.session_state.debug_logs)
            logs.reverse()  # Show most recent first
            for log in logs[:20]:  # Show last 20 logs
                st.text(log)

def validate_api_keys(comparison_model):
    """Validate that all required API keys are present and valid."""
    missing_keys = []
    invalid_keys = []
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    else:
        try:
            client = OpenAI()
            client.models.list()  # Test the API key
        except Exception:
            invalid_keys.append("OPENAI_API_KEY")
    
    # Check W&B API key
    if not os.getenv("WANDB_API_KEY"):
        missing_keys.append("WANDB_API_KEY")
    else:
        try:
            wandb.login()
        except Exception:
            invalid_keys.append("WANDB_API_KEY")
    
    # Check AWS credentials if using Bedrock
    if comparison_model not in openai_models and not comparison_model.startswith("wandb-artifact:///"):
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            missing_keys.append("AWS_ACCESS_KEY_ID")
        if not os.getenv("AWS_SECRET_ACCESS_KEY"):
            missing_keys.append("AWS_SECRET_ACCESS_KEY")
        if not os.getenv("AWS_SESSION_TOKEN"):
            missing_keys.append("AWS_SESSION_TOKEN")
        if not os.getenv("AWS_DEFAULT_REGION"):
            missing_keys.append("AWS_DEFAULT_REGION")
        
        if not any(key in missing_keys for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]):
            try:
                boto3.client(
                    service_name="bedrock-runtime",
                    region_name=os.environ["AWS_DEFAULT_REGION"],
                    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                    aws_session_token=os.environ["AWS_SESSION_TOKEN"],
                )
            except Exception as e:
                print(f"Error initializing bedrock: {e}")
                invalid_keys.extend(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"])
    
    return missing_keys, invalid_keys

if __name__ == "__main__":
    # Setup
    load_dotenv("utils/.env")
    st.set_page_config(page_title="Hiring Assistant", layout="wide")
    st.title("Hiring Assistant")
    
    # Initialize session state
    if 'expert_review_needed' not in st.session_state:
        st.session_state.expert_review_needed = False
    if 'call_id_to_annotate' not in st.session_state:
        st.session_state.call_id_to_annotate = None
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    # Add states for form submission across reloads
    if 'pending_annotation' not in st.session_state:
        st.session_state.pending_annotation = False
    if 'form_decision' not in st.session_state:
        st.session_state.form_decision = None
    if 'form_reason' not in st.session_state:
        st.session_state.form_reason = None
    
    # Create a utility function for logging debug info
    def log_debug(message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        st.session_state.debug_logs.append(log_entry)
        print(log_entry)  # Also print to console for server logs
    
    # Sidebar for mode selection and model configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Add W&B configuration section
        st.subheader("W&B Configuration")
        # Default entity and project from environment variables or hardcoded defaults
        default_entity = os.environ.get("WANDB_ENTITY", "wandb-smle")
        default_project = os.environ.get("WANDB_PROJECT", "e2e-hiring-assistant-test")

        wandb_entity = st.text_input("W&B Entity", value=default_entity, help="Your W&B entity/username")
        wandb_project = st.text_input("W&B Project", value=default_project, help="Your W&B project name")
        
        # Save to session state
        st.session_state.wandb_entity = wandb_entity
        st.session_state.wandb_project = wandb_project
        
        # Initialize Weave client with the provided entity and project
        client = weave.init(f"{wandb_entity}/{wandb_project}")
        
        mode = st.selectbox("Select Mode", ["Create Dataset", "Manage Prompts", "Single Test", "Batch Testing"])
        
        st.subheader("Model Settings")
        extraction_model = st.selectbox("Extraction Model", openai_models)
        comparison_model = st.selectbox(
            "Comparison Model", 
            openai_models+[
                "us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
                "us.amazon.nova-lite-v1:0",
                "custom-wandb-artifact-model"
            ]
        )
        
        # Add API Key Validation section after model selection
        st.subheader("API Key Status")
        missing_keys, invalid_keys = validate_api_keys(comparison_model)
        
        if missing_keys:
            st.error("❌ Missing API Keys:")
            for key in missing_keys:
                st.error(f"- {key}")
            st.warning("Please add the missing keys to your .env file")
        
        if invalid_keys:
            st.error("❌ Invalid API Keys:")
            for key in invalid_keys:
                st.error(f"- {key}")
            st.warning("Please check your API keys in the .env file")
        
        if not missing_keys and not invalid_keys:
            st.success("✅ All required API keys are valid!")
        
        # Initialize AWS client if needed
        if comparison_model not in openai_models and not comparison_model.startswith("wandb-artifact:///"):
            boto_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.environ["AWS_DEFAULT_REGION"],
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                aws_session_token=os.environ["AWS_SESSION_TOKEN"],
            )
            # patching step for Weave
            patch_client(boto_client)
        
        # Add optional input field for custom artifact path
        custom_artifact = st.text_input(
            "Custom W&B Artifact Path (optional)", 
            value="wandb-smle/e2e-hiring-assistant-test/fine-tuned-comparison-model:latest",
            help="Enter a W&B artifact path. Note: 'wandb-artifact:///' will be automatically added as prefix."
        )
        
        # Add button to add new models
        add_model = st.button("+ Add Model to Ollama")
        
        # If custom artifact is provided, use it instead
        if comparison_model == "custom-wandb-artifact-model" and custom_artifact:
            comparison_model = "wandb-artifact:///" + custom_artifact
            
            # Only check for Ollama if using a W&B artifact and the add button is clicked
            if add_model:
                # Extract model name from the artifact path
                artifact_name = comparison_model.split("/")[-1]
                model_name = artifact_name.split(":")[0]
                
                # Create local_models directory if it doesn't exist
                local_models_dir = "local_models"
                os.makedirs(local_models_dir, exist_ok=True)
                
                # Check if the model already exists in Ollama
                try:
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
                    if model_name in result.stdout:
                        print(f"Using existing Ollama model: {model_name}")
                    else:
                        print(f"Model {model_name} not found in Ollama, downloading from W&B...")
                        # Download the artifact from W&B
                        model_dir = os.path.join(local_models_dir, model_name)
                        os.makedirs(model_dir, exist_ok=True)
                        
                        # Extract artifact reference from the path
                        artifact_ref = comparison_model.replace("wandb-artifact:///", "")
                        
                        # Download the artifact
                        artifact = wandb.Api().artifact(artifact_ref)
                        artifact_dir = artifact.download(root=model_dir)
                        
                        # Check if Modelfile exists in the artifact
                        modelfile_path = os.path.join(artifact_dir, "Modelfile")
                        if not os.path.exists(modelfile_path):
                            # If no Modelfile, create one pointing to the root of the artifact folder
                            with open(modelfile_path, "w") as f:
                                f.write(f"FROM {artifact_dir}\n")
                        
                        # Create the Ollama model
                        subprocess.run(
                            ["ollama", "create", model_name, "-f", modelfile_path],
                            check=True
                        )
                        print(f"Created Ollama model: {model_name}")
                except Exception as e:
                    print(f"Error setting up Ollama model: {e}")
                    raise
        
        guardrail_model = st.selectbox("Guardrail Model", openai_models+["smollm2-135m-v18"])
        use_guardrail = st.toggle("Use Guardrail", value=True)
        
        # Expert review settings
        st.subheader("Expert Review Settings")
        expert_review_options = st.radio(
            "Expert Review Mode",
            ["Default (when needed)", "Always On", "Disabled (never)"],
            help="Control when expert review is triggered"
        )
        
        # Map the radio selection to the parameters
        hitl_always_on = expert_review_options == "Always On"
        disable_expert_review = expert_review_options == "Disabled (never)"

        st.subheader("Evaluation Settings")
        judge_model = st.selectbox("Judge Model", ["gpt-4o-mini", "gpt-4o"])

    # Initialize agent with selected configuration
    hiring_agent = HiringAgent(
        extraction_model=extraction_model,
        comparison_model=comparison_model,
        guardrail_model=guardrail_model,
        hitl_always_on=hitl_always_on,
        disable_expert_review=disable_expert_review
    )

    if mode == "Manage Prompts":
        st.header("Prompt Management")
        
        # Define prompts and their descriptions
        prompts = {
            "context_prompt": {
                "name": "Context Prompt",
                "description": "Provides context for the job offer and application comparison",
                "content": context_prompt
            },
            "guardrail_prompt": {
                "name": "Guardrail Prompt",
                "description": "Ensures the model's reasoning is based on provided information",
                "content": guardrail_prompt
            },
            "extract_offer_prompt": {
                "name": "Extract Offer Prompt",
                "description": "Extracts structured information from job offers",
                "content": extract_offer_prompt
            },
            "extract_application_prompt": {
                "name": "Extract Application Prompt",
                "description": "Extracts structured information from applications",
                "content": extract_application_prompt
            },
            "compare_offer_application_prompt": {
                "name": "Compare Offer Application Prompt",
                "description": "Compares job offers and applications to make hiring decisions",
                "content": compare_offer_application_prompt
            }
        }
        
        # Create tabs for each prompt
        tabs = st.tabs(list(prompts.keys()))
        
        for tab, (prompt_id, prompt_info) in zip(tabs, prompts.items()):
            with tab:
                st.subheader(prompt_info["name"])
                st.write(prompt_info["description"])
                
                # Get latest version of the prompt
                try:
                    latest_prompt = weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/{prompt_id}:latest").get()
                    current_content = latest_prompt.content if hasattr(latest_prompt, 'content') else latest_prompt
                except Exception:
                    current_content = prompt_info["content"]
                
                # Create text area for editing
                edited_content = st.text_area(
                    "Edit Prompt",
                    value=current_content,
                    height=400,
                    key=f"edit_{prompt_id}"
                )
                
                # Add publish button
                if st.button(f"Publish {prompt_info['name']}", key=f"publish_{prompt_id}"):
                    try:
                        # Always publish when button is clicked
                        weave_prompt = weave.StringPrompt(edited_content)
                        weave.publish(weave_prompt, name=prompt_id)
                        st.success(f"Successfully published new version of {prompt_info['name']}")
                    except Exception as e:
                        st.error(f"Error publishing prompt: {str(e)}")
                
                # Show version history
                st.subheader("Version History")
                try:
                    versions = weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/{prompt_id}").versions()
                    for version in versions:
                        version_content = version.get().content if hasattr(version.get(), 'content') else version.get()
                        st.text(f"Version: {version.digest[:8]} - {version.created_at}")
                        with st.expander("View Content"):
                            st.text(version_content)
                except Exception:
                    st.info("No version history available")

    elif mode == "Single Test":
        st.header("Upload Documents")
        
        # Check if we're in the middle of an expert review
        if st.session_state.in_expert_review:
            print("Resuming expert review panel...")
            streamlit_expert_panel()
        else:
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
                    # Clear previous results
                    st.session_state.waiting_for_review = False
                    st.session_state.in_expert_review = False
                    st.session_state.call_id_to_annotate = None
                    st.session_state.expert_reason_input = ""
                    st.session_state.expert_decision_input = "True"
                    
                    with st.spinner("Processing documents..."):
                        print("Starting document evaluation...")
                        
                        # Process the documents
                        offer_images = pdf_to_images("temp_offer.pdf")
                        application_images = pdf_to_images("temp_application.pdf")
                        
                        # Run the prediction
                        result = hiring_agent.predict(
                            offer_pdf="temp_offer.pdf",
                            application_pdf="temp_application.pdf",
                            offer_images=offer_images,
                            application_images=application_images
                        )
                        
                        print("Prediction complete. Results saved.")
                    
                    # Check if expert review is needed
                    if st.session_state.waiting_for_review:
                        print("Showing expert review panel...")
                        st.warning("⚠️ Expert Review Required")
                        # Set the flag before showing the panel
                        st.session_state.in_expert_review = True 
                        streamlit_expert_panel()
                    else:
                        # Show normal output
                        print("Showing normal output...")
                        st.header("Results")
                        if result['interview']:
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
            f"weave:///{wandb_entity}/{wandb_project}/object/evaluation_dataset:rERmeHPF4pmNYpTbAayZyFAr3oEzZlhYyVXyhPDI48w",
        )
        trials = st.number_input("Number of Trials", min_value=1, value=1)
        
        if st.button("Run Evaluation"):
            with st.spinner("Running batch evaluation..."):
                benchmark = weave.Evaluation(
                    dataset=weave.ref(dataset_ref).get(),
                    scorers=[decision_match, ReasonScorer(model_id=judge_model)],
                    # replaced lambda to work better with weave versioning
                    preprocess_model_input=pre_process_eval,
                    trials=trials
                )
                results = asyncio.run(benchmark.evaluate(hiring_agent))
                st.json(results)
    # Add this to the "Create Dataset" section in the main function
    else:  # Create Dataset
        st.header("Create Evaluation Dataset")
        
        # Add tabs for the two-step process
        tab1, tab2, tab3 = st.tabs(["1. Generate Characteristics", "2. Calculate R Score", "3. Generate Applications"])
        
        with tab1:
            st.subheader("Step 1: Generate Applicant Characteristics")
            num_applicants = st.number_input("Number of Applicants", min_value=1, value=3, key="num_applicants_step1")
            
            # Upload job offers
            offer_files = st.file_uploader("Upload Offer PDFs", type=['pdf'], accept_multiple_files=True, key="offer_files_step1")
            
            # Bias controls
            st.subheader("Bias Factors (0 = no bias)")
            gender_bias = st.slider("Gender Bias", min_value=-1.0, max_value=1.0, value=0.0, step=0.1, 
                                help="Positive values bias toward male applicants", key="gender_bias_step1")
            age_bias = st.slider("Age Bias", min_value=-1.0, max_value=1.0, value=0.0, step=0.1,
                                help="Positive values bias toward younger applicants", key="age_bias_step1")
            nationality_bias = st.slider("Nationality Bias", min_value=-1.0, max_value=1.0, value=0.0, step=0.1,
                                    help="Positive values bias toward certain nationalities", key="nationality_bias_step1")
            
            bias_factors = {
                "gender": gender_bias,
                "age": age_bias,
                "nationality": nationality_bias
            }
            
            if st.button("Generate Characteristics Table", key="generate_characteristics_btn"):
                if not offer_files:
                    st.error("Please upload at least one job offer PDF")
                else:
                    with st.spinner("Generating applicant characteristics..."):
                        # Create offers directory if it doesn't exist
                        os.makedirs("./utils/data/offers", exist_ok=True)
                        os.makedirs("./utils/data/applications", exist_ok=True)
                        
                        # Save uploaded files and get job position IDs
                        offer_paths = {}
                        for f in offer_files:
                            job_id = f.name.split('.')[0]  # Use filename as job ID
                            save_path = f"./utils/data/offers/{f.name}"
                            if not os.path.exists(save_path):
                                with open(save_path, "wb") as file:
                                    file.write(f.getvalue())
                            offer_paths[job_id] = save_path
                        
                        # Generate characteristics table
                        characteristics_df = generate_applicant_characteristics(
                            num_applicants=num_applicants,
                            job_positions=list(offer_paths.keys()),
                            bias_factors=bias_factors
                        )
                        
                        # Save to session state for next step
                        st.session_state.characteristics_df = characteristics_df
                        st.session_state.offer_paths = offer_paths
                        
                        # Log to W&B as artifact
                        run = wandb.init(project=wandb_project, entity=wandb_entity, job_type="dataset-generation")
                        
                        # Convert DataFrame to wandb.Table
                        characteristics_table = wandb.Table(dataframe=characteristics_df)
                        
                        # Create and log artifact
                        artifact = wandb.Artifact(
                            name="applicant_characteristics",
                            type="dataset",
                            description="Structured table of applicant characteristics"
                        )
                        artifact.add(characteristics_table, "characteristics")
                        run.log_artifact(artifact)
                        
                        # Store artifact path in session state
                        artifact_path = f"{run.entity}/{run.project}/{artifact.name}:latest"
                        st.session_state.characteristics_artifact_path = artifact_path
                        
                        # Save as Weave dataset too
                        rows = characteristics_df.to_dict('records')
                        weave_dataset = weave.Dataset(name="applicant_characteristics", rows=rows)
                        weave_ref = weave.publish(weave_dataset)
                        weave_url = urls.object_version_path(
                            weave_ref.entity,
                            weave_ref.project,
                            weave_ref.name,
                            weave_ref.digest,
                        )
                        
                        # Store Weave reference in session state
                        st.session_state.characteristics_weave_ref = weave_url
                        
                        # Finish the run
                        wandb.finish()
                        
                        # Display the table
                        st.dataframe(characteristics_df)
                        
                        # Display Weave dataset link
                        st.success(f"Characteristics table generated successfully! [View Weave dataset]({weave_url})")
                        
                        # Move to next tab
                        st.success("Proceed to Step 2.")
        
        with tab2:
            st.subheader("Step 2: Calculate R Score")
            
            # Input for artifact path
            artifact_path = st.text_input(
                "Characteristics Artifact Path", 
                value=st.session_state.get("characteristics_artifact_path", "wandb-smle/e2e-hiring-assistant/applicant_characteristics:latest"),
                help="Format: entity/project/artifact_name:version",
                key="artifact_path_step2"
            )
            
            if st.button("Calculate R Score", key="calculate_r_score_btn"):
                with st.spinner("Calculating R Score..."):
                    # Initialize W&B run
                    run = wandb.init(project=wandb_project, entity=wandb_entity, job_type="dataset-evaluation")
                    
                    # Use the artifact from the provided path
                    artifact = run.use_artifact(artifact_path)
                    characteristics_table = artifact.get("characteristics")
                    
                    # Convert wandb.Table to DataFrame
                    df = characteristics_table.get_dataframe()
                    
                    # Store in session state for next step
                    st.session_state.characteristics_df = df
                    
                    # Calculate R score
                    r_score = calculate_r_score(df)
                    
                    # Save R score to session state
                    st.session_state.r_score = r_score
                    
                    # Display R score with color coding
                    if r_score >= 0.8:
                        st.success(f"R Score: {r_score:.2f} - Excellent representativeness")
                    elif r_score >= 0.6:
                        st.info(f"R Score: {r_score:.2f} - Good representativeness")
                    else:
                        st.warning(f"R Score: {r_score:.2f} - Poor representativeness, consider regenerating")
                    
                    # Show detailed metrics
                    st.subheader("Dataset Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Gender Distribution")
                        st.bar_chart(df['gender'].value_counts())
                        
                        st.write("Age Distribution")
                        hist_values = np.histogram(df['age'], bins=8, range=(25, 65))[0]
                        st.bar_chart(hist_values)
                    
                    with col2:
                        st.write("Nationality Distribution")
                        st.bar_chart(df['nationality'].value_counts())
                        
                        st.write("Positive/Negative Examples")
                        st.bar_chart(df['is_positive'].map({True: "Positive", False: "Negative"}).value_counts())
                    
                    # Log R score to W&B
                    wandb.log({"r_score": r_score})
                    
                    # Finish the run
                    wandb.finish()
                    
                    st.success("R Score calculated successfully! Proceed to Step 3.")
        
        with tab3:
            st.subheader("Step 3: Generate Applications")
            
            # Check for offer PDFs first
            if 'offer_paths' not in st.session_state:
                st.warning("Please upload job offer files before proceeding.")
                offer_uploader = st.file_uploader("Upload Offer PDFs", type=['pdf'], accept_multiple_files=True, key="offer_files_step3")
                
                if offer_uploader:
                    # Save uploaded files and get job position IDs
                    offer_paths = {}
                    for f in offer_uploader:
                        job_id = f.name.split('.')[0]  # Use filename as job ID
                        save_path = f"./utils/data/offers/{f.name}"
                        if not os.path.exists(save_path):
                            with open(save_path, "wb") as file:
                                file.write(f.getvalue())
                        offer_paths[job_id] = save_path
                    st.session_state.offer_paths = offer_paths
                    st.success("Job offer files uploaded successfully!")
                else:
                    st.error("Cannot proceed without job offer files")
                    st.stop()

            # Input for characteristics artifact path
            characteristics_artifact_path = st.text_input(
                "Characteristics Artifact Path",
                value=st.session_state.get("characteristics_artifact_path", "wandb-smle/e2e-hiring-assistant/applicant_characteristics:latest"),
                help="Format: entity/project/artifact_name:version",
                key="artifact_path_step3"
            )
            
            # Generation model selection
            generation_model = st.selectbox("Generation Model", openai_models, key="generation_model_step3")
            
            # Show R score threshold control
            r_threshold = st.slider("R Score Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                                help="Minimum acceptable R score to proceed with generation", key="r_threshold_step3")
            
            if st.button("Generate Applications", key="generate_applications_btn"):
                with st.spinner("Loading characteristics and generating applications..."):
                    # Initialize a new W&B run for application generation
                    run = wandb.init(project=wandb_project, entity=wandb_entity, job_type="application-generation")
                    
                    # Download the characteristics artifact
                    artifact = run.use_artifact(characteristics_artifact_path)
                    characteristics_table = artifact.get("characteristics")
                    
                    # Convert wandb.Table to DataFrame
                    df = characteristics_table.get_dataframe()
                    
                    # Calculate R score for the loaded characteristics
                    r_score = calculate_r_score(df)
                    
                    # Show current R score
                    if r_score >= r_threshold:
                        st.success(f"Current R Score: {r_score:.2f} (above threshold)")
                        can_proceed = True
                    else:
                        st.error(f"Current R Score: {r_score:.2f} (below threshold)")
                        st.warning("Consider regenerating the characteristics table with less bias to improve the R score.")
                        can_proceed = False
                        
                    if can_proceed:
                        # Generate applications using offer paths from session state
                        dataset_struct = generate_application_from_characteristics(
                            characteristics_df=df,
                            offer_paths=st.session_state.offer_paths,
                            generation_model=generation_model
                        )
                        
                        # Convert to rows for Weave dataset
                        rows = [example.model_dump() for example in dataset_struct.examples]
                        
                        # Create and publish Weave dataset
                        dataset = weave.Dataset(name="evaluation_dataset", rows=rows)
                        weave_ref = weave.publish(dataset)
                        weave_url = urls.object_version_path(
                            weave_ref.entity,
                            weave_ref.project,
                            weave_ref.name,
                            weave_ref.digest,
                        )
                        
                        # Create a wandb.Table from the dataset rows
                        columns = list(rows[0].keys()) if rows else []
                        table = wandb.Table(columns=columns)
                        for row in rows:
                            table.add_data(*[row[col] for col in columns])
                        
                        # Create and log artifact with the table
                        artifact = wandb.Artifact(
                            name="evaluation_dataset",
                            type="dataset",
                            description="Generated applications based on characteristics"
                        )
                        artifact.add(table, "evaluation_dataset")
                        
                        # Add metadata
                        artifact.metadata = {
                            "r_score": r_score,
                            "num_examples": len(rows),
                            "positive_examples": sum(row["interview"] for row in rows),
                            "negative_examples": sum(not row["interview"] for row in rows),
                            "weave_reference": weave_url,
                            "characteristics_source": characteristics_artifact_path
                        }
                        
                        # Log artifact
                        run.log_artifact(artifact)
                        
                        # Finish the run
                        wandb.finish()
                        
                        # Display success message with links
                        st.success(f"Dataset created and published successfully!")
                        st.markdown(f"[View Weave dataset]({weave_url})")
                        st.markdown(f"[View W&B artifact](https://wandb.ai/{run.entity}/{run.project}/artifacts/{artifact.type}/{artifact.name}/v0)")
                        
                        # Display sample of generated applications
                        st.subheader("Sample of Generated Applications")
                        for i, example in enumerate(dataset_struct.examples[:5]):  # Show first 5 examples
                            with st.expander(f"Example {i+1} - {'Positive' if example.interview else 'Negative'}"):
                                st.write("**Job Offer:**")
                                st.write(example.offer_text[:500] + "...")  # Show first 500 chars
                                
                                st.write("**Application:**")
                                st.write(example.application_text[:500] + "...")  # Show first 500 chars
                                
                                st.write("**Reason:**")
                                st.write(example.reason)