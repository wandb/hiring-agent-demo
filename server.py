# server.py
import weave
from weave.scorers import HallucinationFreeScorer
from weave.scorers.hallucination_scorer import HallucinationResponse
from weave.trace.api import get_current_call
from weave.trace import urls
from weave.integrations.bedrock.bedrock_sdk import patch_client
from weave.flow.annotation_spec import AnnotationSpec
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI

import boto3
import os, sys, json, subprocess, shutil
import asyncio, uuid, nest_asyncio
import gc

from PIL import Image
from typing import Optional, List, Any, Union
from typing_extensions import TypedDict
from pydantic import PrivateAttr
import streamlit as st

import pandas as pd
import numpy as np
import wandb
import time
import resource
import openai
import botocore
from langchain_core.exceptions import LangChainException

from utils.prompt import (
    CV, Offer, InterviewDecision, 
    context_prompt, guardrail_prompt,
    extract_offer_prompt, extract_application_prompt, 
    compare_offer_application_prompt, reason_comp_prompt,
    ReasonComparison)
from utils.prepro import extract_text_from_pdf, pdf_to_images, pre_process_eval
from utils.evaluate import DecisionScorer, ReasonScorer
from utils.generate import generate_dataset, generate_applicant_characteristics, calculate_r_score, generate_application_from_characteristics
from dotenv import load_dotenv

# Apply nest_asyncio at the module level to allow nested event loops
nest_asyncio.apply()

# Initialize Streamlit session state variables
if 'thread_config' not in st.session_state:
    st.session_state.thread_config = {"configurable": {"thread_id": uuid.uuid4()}}
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

@weave.op
def reset_state(state: GraphState) -> GraphState:
    """Reset the state's counter fields to ensure a fresh start for each prediction."""
    # Create a new state dictionary with initial values
    # Keep the input PDFs but reset all other values
    reset_state = {
        "offer_pdf": state.get("offer_pdf", ""),
        "application_pdf": state.get("application_pdf", ""),
        "job_offer_extract": "",
        "application_extract": "",
        "reason": "",
        "has_hallucination": False,
        "decision": False,
        "tries": 0,  # Explicitly reset the tries counter
        "last_comparison": None
    }
    print("State has been reset for new prediction")
    return reset_state

class ExtractJobOffer:
    def __init__(self, extraction_client, extract_offer_prompt) -> None:
        self.extraction_client = extraction_client
        self.extract_offer_prompt = extract_offer_prompt

    @weave.op(name="ExtractJobOfferCall")
    def __call__(self, state: GraphState):
        """Extract the information from the job offer"""
        
        job_offer = extract_text_from_pdf(state["offer_pdf"])
        job_offer_extract = self.extraction_client.invoke(self.extract_offer_prompt.format(job_offer=job_offer))
        state["job_offer_extract"] = job_offer_extract.content
        return state

class ExtractApplication:
    def __init__(self, extraction_client, extract_application_prompt) -> None:
        self.extraction_client = extraction_client
        self.extract_application_prompt = extract_application_prompt

    @weave.op(name="ExtractApplicationCall")
    def __call__(self, state: GraphState):
        """Extract the information from the application"""
        
        application = extract_text_from_pdf(state["application_pdf"])
        application_extract = self.extraction_client.invoke(self.extract_application_prompt.format(application=application))
        state["application_extract"] = application_extract.content
        return state

class CompareApplicationOffer:
    def __init__(self, comparison_client, compare_offer_application_prompt) -> None:
        self.comparison_client = comparison_client
        self.compare_offer_application_prompt = compare_offer_application_prompt

    @weave.op(name="CompareApplicationOfferCall")
    def __call__(self, state: GraphState):
        """Compare the application and offer and decide if they are fitting and why."""
        # TODO: should wandb be used to track in inference? (would need to make work with weave parallel)
        # track inference
        # run = wandb.init(
        #     project=wandb_project, 
        #     entity=wandb_entity, 
        #     job_type="inference"
        # )

        state["tries"] = 1 if not state.get("tries") else state.get("tries")+1
        application_extract = state["application_extract"]
        job_offer_extract = state["job_offer_extract"]
        
        comparison_document = self.compare_offer_application_prompt.format(
            job_offer_extract=job_offer_extract, 
            application_extract=application_extract
        )
        
        decision = self.comparison_client.invoke(comparison_document)
        state["reason"] = decision.reason
        state["decision"] = decision.decision


        # run.log({
        #     "decision": decision.decision,
        #     "reason": decision.reason,
        #     "job_offer_extract": job_offer_extract,
        #     "application_extract": application_extract,
        #     "comparison_document": comparison_document
        # })
        # run.finish()
        return state

class HallucinationGuardrail:
    def __init__(self, guardrail_client, context_prompt, guardrail_prompt) -> None:
        self.guardrail_client = guardrail_client
        self.context_prompt = context_prompt
        self.guardrail_prompt = guardrail_prompt

    @weave.op(name="HallucinationGuardrail")
    def __call__(self, state: GraphState):
        """Use guardrail to check whether reason only contains info from application or offer"""
        application_extract = state["application_extract"]
        job_offer_extract = state["job_offer_extract"]
        
        decision_reason = "Decision: We should move on with an interview\n" if state["decision"] else "Decision: We should NOT move on with an interview\n"
        decision_reason += f"Reason: {state['reason']}"
        
        context_document = self.context_prompt.format(
            job_offer_extract=job_offer_extract,
            application_extract=application_extract
        )

        # With nest_asyncio already applied, we can safely use asyncio.run
        guardrail_result = asyncio.run(self.guardrail_client.score(
            output=decision_reason, 
            context=context_document
        ))
        
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
        # Get call from streamlit session state if it exists, otherwise get current call
        if st.session_state.call_id_to_annotate:
            call = client.get_call(st.session_state.call_id_to_annotate)
            print(f"Found call {call.id} in streamlit session state")
        else:
            call = get_current_call()
            print(f"Found call {call.id} in current call")
        
        # Add feedback if we have a valid call
        if call:
            call.feedback.add_reaction("🤖")
        else:
            st.error("Could not find a valid call to add feedback to")

        return "retry"
    elif has_hallucination == True and state["tries"] >= 2:
        print("---DECISION: Hallucination detected, too many retries, marking for expert review---")
        return "expert_review"
    else:
        print("---DECISION: Answer Accepted---")
        return "valid"

def create_wf(
        extraction_client,
        comparison_client,
        guardrail_client,
        hitl_always_on: bool,
        extract_offer_prompt,
        extract_application_prompt,
        compare_offer_application_prompt,
        context_prompt,
        guardrail_prompt,
        disable_expert_review: bool = False):
    # Define the nodes in the workflow
    workflow = StateGraph(GraphState)
    
    # Add a reset state node at the beginning
    workflow.add_node("reset_state", reset_state)
    workflow.add_node("extract_job_offer", ExtractJobOffer(extraction_client=extraction_client, extract_offer_prompt=extract_offer_prompt))
    workflow.add_node("extract_application", ExtractApplication(extraction_client=extraction_client, extract_application_prompt=extract_application_prompt))
    workflow.add_node("compare_application_offer", CompareApplicationOffer(comparison_client=comparison_client, compare_offer_application_prompt=compare_offer_application_prompt))
    workflow.add_node("hallucination_guardrail", HallucinationGuardrail(guardrail_client=guardrail_client, context_prompt=context_prompt, guardrail_prompt=guardrail_prompt))
    workflow.add_node("expert_review", expert_review)

    # Start from 'reset_state' step
    workflow.add_edge(START, "reset_state")
    workflow.add_edge("reset_state", "extract_job_offer")
    workflow.add_edge("extract_job_offer", "extract_application")
    workflow.add_edge("extract_application", "compare_application_offer")
    workflow.add_edge("compare_application_offer", "hallucination_guardrail")
    
    # Configure the flow based on expert review settings
    if disable_expert_review:
        # Never use expert review, even for hallucinations
        # But still use retry logic for hallucinations
        workflow.add_conditional_edges(
            "hallucination_guardrail",
            validate,
            {
                "retry": "compare_application_offer",
                "expert_review": END,  # Skip expert review but keep retry
                "valid": END,
            },
        )
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

        # Add emoji showing expert review
        call.feedback.add_reaction("👀")

        spec1 = AnnotationSpec(
            name="Expert Decision",
            description="Expert hiring decisions and reasoning",
            field_schema={
                "type": "boolean",
              #  "enum": ["True", "False"],
            }
            )
        

        spec2 = AnnotationSpec(
            name="Expert Reason",
            description="Expert reasoning for hiring decisions",
            field_schema={
                "type": "string",
            }
            )
        
        published_decision_spec = weave.publish(spec1, "expert_decision")
        published_reason_spec = weave.publish(spec2, "expert_reason")

        # Add feedback using the annotation spec
        call.feedback.add(
            feedback_type="wandb.annotation." + published_decision_spec.name,
            payload={
                "value": decision_bool,
            },
            annotation_ref=published_decision_spec.uri()
        )

        call.feedback.add(
            feedback_type="wandb.annotation." + published_reason_spec.name,
            payload={
                "value": reason,
            },
            annotation_ref=published_reason_spec.uri()
        )
        
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
    # Generate a new thread ID for each graph run to ensure fresh state
    thread_config = {"configurable": {"thread_id": uuid.uuid4()}}
    st.session_state.thread_config = thread_config
    print(f"Generated new thread ID: {thread_config['configurable']['thread_id']}")

    # Start a new event loop
    result = {}
    print("Starting HiringAgent prediction workflow...")
    
    try:
        # Execute the prediction
        for event in app.stream({"offer_pdf": offer_pdf, "application_pdf": application_pdf}, config=thread_config):
            for key, value in event.items():
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
    context_prompt: weave.StringPrompt
    guardrail_prompt: weave.StringPrompt
    extract_offer_prompt: weave.StringPrompt
    extract_application_prompt: weave.StringPrompt
    compare_offer_application_prompt: weave.StringPrompt
    _app : Any = PrivateAttr()
    _extraction_client = PrivateAttr()
    _comparison_client = PrivateAttr()
    _guardrail_client = PrivateAttr()

    # Start streaming the graph updates using job_offer and cv as inputs
    def model_post_init(self, __context):
        # Initialize clients once and reuse
        self._extraction_client = ChatOpenAI(model=self.extraction_model, max_retries=5)
        
        if self.comparison_model in openai_models:
            self._comparison_client = ChatOpenAI(
                model=self.comparison_model,
                response_format={"type": "json"},
            ).with_structured_output(InterviewDecision)
        elif self.comparison_model.startswith("wandb-artifact:///"):
            # Set Ollama parallelism to match Weave parallelism
            weave_parallelism = int(os.environ.get("WEAVE_PARALLELISM", "1"))
            os.environ["OLLAMA_NUM_PARALLEL"] = str(weave_parallelism)
            
            # Extract full artifact path and convert to Ollama-compatible name
            full_artifact_path = self.comparison_model.replace("wandb-artifact:///", "")
            model_name = full_artifact_path.replace("/", "-").replace(":", "-")
            
            self._comparison_client = ChatOllama(
                model=model_name,
                format="json",
                num_predict=2500,  # Maximum tokens to generate (default: 128)
                #temperature=0.2,   # Lower temperature for more deterministic outputs
                repeat_penalty=1.5,  # Higher penalty to avoid repetition loops
                stop=["}"]         # Stop token to help ensure JSON completion
            ).with_structured_output(InterviewDecision)
        else:
            self._comparison_client = ChatBedrock(
                model=self.comparison_model, 
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                aws_session_token=os.environ["AWS_SESSION_TOKEN"]
            ).with_structured_output(InterviewDecision)
                
        # Initialize hallucination scorer client
        self._guardrail_client = HallucinationFreeScorer(
            model_id=f"openai/{self.guardrail_model}"
        )
                
        self._app = create_wf(
            self._extraction_client, 
            self._comparison_client,
            self._guardrail_client,
            self.hitl_always_on,
            self.extract_offer_prompt,
            self.extract_application_prompt,
            self.compare_offer_application_prompt,
            self.context_prompt,
            self.guardrail_prompt,
            self.disable_expert_review
        )

    @weave.op
    def predict(self, offer_pdf: str, application_pdf: str,
                offer_images: Optional[Union[List[Image.Image], Image.Image]] = None,
                application_images: Optional[Union[List[Image.Image], Image.Image]] = None) -> dict:
        # Get the parent call to add annotations later
        parent_call = get_current_call()
        if parent_call:
            st.session_state.call_id_to_annotate = parent_call.id
        
        # We'll use the existing workflow instance but with a unique thread_id
        # to ensure state isolation between runs
        result = stream_graph_updates(self._app, offer_pdf, application_pdf)
        return result
    
    # Add cleanup method
    def cleanup(self):
        # Close any clients that have close methods
        for client_attr in ['_extraction_client', '_comparison_client', '_guardrail_client']:
            client = getattr(self, client_attr, None)
            if client and hasattr(client, 'close'):
                client.close()

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

# Function to check if prompts exist in Weave
def check_prompt_exists(prompt_id, entity, project):
    """Check if a prompt exists in Weave and return True if it does, False otherwise."""
    try:
        weave.ref(f"weave:///{entity}/{project}/object/{prompt_id}:latest").get()
        return True
    except Exception:
        return False

if __name__ == "__main__":
    # Setup
    load_dotenv("utils/.env")
    st.set_page_config(page_title="Hiring Assistant", layout="wide")
    st.title("Hiring Assistant")
    
    # Ensure all prompts are published to Weave
    try:
        # Get default entity and project
        default_entity = os.environ.get("WANDB_ENTITY", "wandb-smle")
        default_project = os.environ.get("WANDB_PROJECT", "e2e-hiring-assistant-test")
        
        # Initialize Weave client early with default entity/project
        client = weave.init(f"{default_entity}/{default_project}")
    except Exception as e:
        print(f"Error initializing Weave client: {str(e)}")
    
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
                # Extract full artifact path (without the wandb-artifact:/// prefix)
                full_artifact_path = comparison_model.replace("wandb-artifact:///", "")
                # Convert to Ollama-compatible name (replace / and : with -)
                model_name = full_artifact_path.replace("/", "-").replace(":", "-")
                
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

        # Add Performance Settings section
        st.subheader("Performance Settings")
        
        # Get current WEAVE_PARALLELISM value or default to 1
        current_parallelism = os.environ.get("WEAVE_PARALLELISM", "1")
        
        # Add number input widget for WEAVE_PARALLELISM
        weave_parallelism = st.number_input(
            "Weave Parallelism", 
            min_value=1, 
            max_value=16, 
            value=int(current_parallelism),
            help="Controls how many operations Weave can run in parallel. Lower values reduce file handle usage."
        )
        
        # Set environment variable based on user input
        os.environ["WEAVE_PARALLELISM"] = str(weave_parallelism)
        
        # Show info about current setting
        if weave_parallelism <= 2:
            st.info("Low parallelism: Reduces 'Too many open files' errors but may be slower")
        elif weave_parallelism >= 8:
            st.warning("High parallelism: Faster but may cause 'Too many open files' errors on large datasets")

        # Add system ulimit control
        st.subheader("System Resource Settings")
        
        # Add a button to increase system file limit
        if st.button("Increase System File Limit"):
            try:
                # Try to set higher soft limit for the current process (works on Unix systems)
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft * 2, hard), hard))
                new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                st.success(f"Increased file limit from {soft} to {new_soft} (max: {hard})")
            except Exception as e:
                st.error(f"Failed to increase file limit: {str(e)}")
                st.info("Try running 'ulimit -n 4096' in your terminal before starting the app")
        
        # Display current file limit information
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            st.info(f"Current file limit: {soft} (max: {hard})")
        except:
            st.info("File limit information not available")

    # Initialize agent with selected configuration
    # First check if all required prompts exist in Weave
    required_prompts = [
        "context_prompt", 
        "guardrail_prompt", 
        "extract_offer_prompt", 
        "extract_application_prompt", 
        "compare_offer_application_prompt"
    ]
    
    missing_prompts = []
    for prompt_id in required_prompts:
        if not check_prompt_exists(prompt_id, wandb_entity, wandb_project):
            missing_prompts.append(prompt_id)
    
    if missing_prompts and mode != "Manage Prompts":
        prompt_list = ", ".join(missing_prompts)
        st.error(f"⚠️ Missing prompts in Weave: {prompt_list}")
        st.warning("Please go to 'Manage Prompts' tab and publish all required prompts before using the application.")
        st.stop()  # Stop execution if prompts are missing and not in Manage Prompts mode
        
    hiring_agent = HiringAgent(
        extraction_model=extraction_model,
        comparison_model=comparison_model,
        guardrail_model=guardrail_model,
        hitl_always_on=hitl_always_on,
        disable_expert_review=disable_expert_review,
        context_prompt=weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/context_prompt:latest").get(),
        guardrail_prompt=weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/guardrail_prompt:latest").get(),
        extract_offer_prompt=weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/extract_offer_prompt:latest").get(),
        extract_application_prompt=weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/extract_application_prompt:latest").get(),
        compare_offer_application_prompt=weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/compare_offer_application_prompt:latest").get()
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
            },
            "reason_comp_prompt": {
                "name": "Reason Comparison Prompt",
                "description": "Used for evaluating the reasoning of the decision model",
                "content": reason_comp_prompt
            }
        }
        
        # Display overall prompt status
        st.subheader("Prompt Status")
        status_cols = st.columns(len(prompts))
        
        for i, (prompt_id, prompt_info) in enumerate(prompts.items()):
            exists = check_prompt_exists(prompt_id, wandb_entity, wandb_project)
            with status_cols[i]:
                if exists:
                    st.success(f"{prompt_info['name']}: ✅")
                else:
                    st.error(f"{prompt_info['name']}: ❌")
        
        st.write("---")
        
        # Create tabs for each prompt
        tabs = st.tabs(list(prompts.keys()))
        
        for tab, (prompt_id, prompt_info) in zip(tabs, prompts.items()):
            with tab:
                # Check if this prompt exists
                exists = check_prompt_exists(prompt_id, wandb_entity, wandb_project)
                
                st.subheader(prompt_info["name"])
                st.write(prompt_info["description"])
                
                if not exists:
                    st.warning(f"⚠️ This prompt has not been published yet. Click 'Publish {prompt_info['name']}' below to make it available to the application.")
                
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
            f"weave:///{wandb_entity}/{wandb_project}/object/evaluation_dataset:latest",
            help="Use the evaluation dataset, not the finetuning dataset"
        )
        trials = st.number_input("Number of Trials", min_value=1, value=1)
        
        # Automatically disable expert review for batch testing
        disable_expert_review = True
        
        # Show information about expert review mode
        st.info("Expert review is automatically disabled during batch testing, but retry logic remains active.")
        
        if st.button("Run Evaluation"):
            with st.spinner("Running batch evaluation..."):
                try:
                    # Check if reason_comp_prompt exists in Weave
                    if not check_prompt_exists("reason_comp_prompt", wandb_entity, wandb_project):
                        st.error("⚠️ The 'reason_comp_prompt' is missing in Weave")
                        st.warning("Please go to 'Manage Prompts' tab and publish the 'Reason Comparison Prompt' before running batch evaluation.")
                        st.stop()  # Stop execution if reason_comp_prompt is missing
                    
                    # Create a dedicated batch agent only when running the evaluation
                    # This avoids creating a separate agent just for configuration
                    batch_hiring_agent = HiringAgent(
                        extraction_model=extraction_model,
                        comparison_model=comparison_model,
                        guardrail_model=guardrail_model,
                        hitl_always_on=False,
                        disable_expert_review=True,
                        context_prompt=weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/context_prompt:latest").get(),
                        guardrail_prompt=weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/guardrail_prompt:latest").get(),
                        extract_offer_prompt=weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/extract_offer_prompt:latest").get(),
                        extract_application_prompt=weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/extract_application_prompt:latest").get(),
                        compare_offer_application_prompt=weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/compare_offer_application_prompt:latest").get()
                    )
                    
                    # Create evaluation setup
                    # Get reason_comp_prompt from Weave (already checked that it exists)
                    reason_prompt = weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/reason_comp_prompt:latest").get()
                    reason_scorer = ReasonScorer(model_id=judge_model, reason_comp_prompt=reason_prompt)
                    
                    benchmark = weave.Evaluation(
                        dataset=weave.ref(dataset_ref).get(),
                        scorers=[DecisionScorer(), reason_scorer],
                        # replaced lambda to work better with weave versioning
                        preprocess_model_input=pre_process_eval,
                        trials=trials
                    )
                    results = asyncio.run(benchmark.evaluate(batch_hiring_agent))
                    st.json(results)
                    
                    # Explicitly clean up resources after batch testing
                    st.info("Cleaning up resources...")
                    batch_hiring_agent.cleanup()
                    
                    # Force garbage collection to release file descriptors
                    collected = gc.collect()
                    st.info(f"Resource cleanup complete. Released {collected} objects.")
                    
                except Exception as e:
                    st.error(f"Batch evaluation error: {str(e)}")
                    # Still attempt cleanup if there was an error
                    if 'batch_hiring_agent' in locals():
                        batch_hiring_agent.cleanup()
                        gc.collect()
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
                    
                    # Create and log data quality plots using native wandb plots
                    
                    # 1. Gender Distribution
                    gender_counts = df['gender'].value_counts().reset_index()
                    gender_counts.columns = ['gender', 'count']
                    gender_table = wandb.Table(dataframe=gender_counts)
                    gender_plot = wandb.plot.bar(
                        gender_table, 
                        "gender", 
                        "count", 
                        title="Gender Distribution"
                    )
                    
                    # 2. Age Distribution
                    age_data = [[age] for age in df['age']]
                    age_table = wandb.Table(data=age_data, columns=["age"])
                    age_plot = wandb.plot.histogram(
                        age_table, 
                        "age", 
                        title="Age Distribution"
                    )
                    
                    # 3. Nationality Distribution
                    nationality_counts = df['nationality'].value_counts().reset_index()
                    nationality_counts.columns = ['nationality', 'count']
                    nationality_table = wandb.Table(dataframe=nationality_counts)
                    nationality_plot = wandb.plot.bar(
                        nationality_table, 
                        "nationality", 
                        "count", 
                        title="Nationality Distribution"
                    )
                    
                    # 4. Positive/Negative Examples
                    example_counts = df['is_positive'].map({True: "Positive", False: "Negative"}).value_counts().reset_index()
                    example_counts.columns = ['decision', 'count']
                    example_table = wandb.Table(dataframe=example_counts)
                    example_plot = wandb.plot.bar(
                        example_table, 
                        "decision", 
                        "count", 
                        title="Positive/Negative Examples"
                    )
                    
                    # 5. Education Distribution
                    education_counts = df['education'].value_counts().reset_index()
                    education_counts.columns = ['education', 'count']
                    education_table = wandb.Table(dataframe=education_counts)
                    education_plot = wandb.plot.bar(
                        education_table, 
                        "education", 
                        "count", 
                        title="Education Distribution"
                    )
                    
                    # 6. Years of Experience Distribution
                    experience_data = [[exp] for exp in df['years_experience']]
                    experience_table = wandb.Table(data=experience_data, columns=["years_experience"])
                    experience_plot = wandb.plot.histogram(
                        experience_table, 
                        "years_experience", 
                        title="Years of Experience Distribution"
                    )
                    
                    # 7. Quality Score Distribution
                    quality_data = [[score] for score in df['quality_score']]
                    quality_table = wandb.Table(data=quality_data, columns=["quality_score"])
                    quality_plot = wandb.plot.histogram(
                        quality_table, 
                        "quality_score", 
                        title="Quality Score Distribution"
                    )
                    
                    # 8. Job Position Distribution
                    position_counts = df['job_position'].value_counts().reset_index()
                    position_counts.columns = ['job_position', 'count']
                    position_table = wandb.Table(dataframe=position_counts)
                    position_plot = wandb.plot.bar(
                        position_table, 
                        "job_position", 
                        "count", 
                        title="Job Position Distribution"
                    )
                    
                    # 9. Age vs Experience Scatter Plot
                    age_exp_data = [[row['age'], row['years_experience']] for _, row in df.iterrows()]
                    age_exp_table = wandb.Table(data=age_exp_data, columns=["age", "years_experience"])
                    age_exp_plot = wandb.plot.scatter(
                        age_exp_table,
                        "age",
                        "years_experience",
                        title="Age vs Experience"
                    )
                    
                    # 10. Quality Score by Gender
                    quality_by_gender = [[row['gender'], row['quality_score']] for _, row in df.iterrows()]
                    quality_gender_table = wandb.Table(data=quality_by_gender, columns=["gender", "quality_score"])
                    quality_gender_plot = wandb.plot.scatter(
                        quality_gender_table,
                        "gender",
                        "quality_score",
                        title="Quality Score by Gender"
                    )
                    
                    # Log all plots to W&B
                    wandb.log({
                        "r_score": r_score,
                        "gender_distribution": gender_plot,
                        "age_distribution": age_plot,
                        "nationality_distribution": nationality_plot,
                        "example_distribution": example_plot,
                        "education_distribution": education_plot,
                        "experience_distribution": experience_plot,
                        "quality_score_distribution": quality_plot,
                        "job_position_distribution": position_plot,
                        "age_vs_experience": age_exp_plot,
                        "quality_score_by_gender": quality_gender_plot
                    })
                    
                    # Show detailed metrics in Streamlit UI
                    st.subheader("Dataset Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Gender Distribution")
                        st.bar_chart(df['gender'].value_counts())
                        
                        st.write("Age Distribution")
                        hist_values = np.histogram(df['age'], bins=8, range=(25, 65))[0]
                        st.bar_chart(hist_values)
                        
                        st.write("Education Distribution")
                        st.bar_chart(df['education'].value_counts())
                    
                    with col2:
                        st.write("Nationality Distribution")
                        st.bar_chart(df['nationality'].value_counts())
                        
                        st.write("Positive/Negative Examples")
                        st.bar_chart(df['is_positive'].map({True: "Positive", False: "Negative"}).value_counts())
                        
                        st.write("Years of Experience")
                        exp_values = np.histogram(df['years_experience'], bins=10)[0]
                        st.bar_chart(exp_values)
                    
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
            
            # Add split ratio control
            split_ratio = st.slider("Train/Eval Split Ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05,
                                help="Percentage of data to use for fine-tuning dataset", key="split_ratio_step3")
            
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
                        
                        # Split the rows into training and evaluation sets
                        import random
                        random.shuffle(rows)
                        split_idx = int(len(rows) * split_ratio)
                        training_rows = rows[:split_idx]
                        evaluation_rows = rows[split_idx:]
                        
                        # Create and publish training Weave dataset
                        training_dataset = weave.Dataset(name="finetuning_dataset", rows=training_rows)
                        training_ref = weave.publish(training_dataset)
                        training_url = urls.object_version_path(
                            training_ref.entity,
                            training_ref.project,
                            training_ref.name,
                            training_ref.digest,
                        )
                        
                        # Create and publish evaluation Weave dataset
                        evaluation_dataset = weave.Dataset(name="evaluation_dataset", rows=evaluation_rows)
                        evaluation_ref = weave.publish(evaluation_dataset)
                        evaluation_url = urls.object_version_path(
                            evaluation_ref.entity,
                            evaluation_ref.project,
                            evaluation_ref.name,
                            evaluation_ref.digest,
                        )
                        
                        # Create wandb.Tables from the dataset rows
                        columns = list(rows[0].keys()) if rows else []
                        
                        # Training table
                        training_table = wandb.Table(columns=columns)
                        for row in training_rows:
                            training_table.add_data(*[row[col] for col in columns])
                        
                        # Evaluation table
                        evaluation_table = wandb.Table(columns=columns)
                        for row in evaluation_rows:
                            evaluation_table.add_data(*[row[col] for col in columns])
                        
                        # Create and log training artifact
                        training_artifact = wandb.Artifact(
                            name="finetuning_dataset",
                            type="dataset",
                            description="Generated applications for fine-tuning"
                        )
                        training_artifact.add(training_table, "finetuning_dataset")
                        
                        # Add metadata to training artifact
                        training_artifact.metadata = {
                            "r_score": r_score,
                            "num_examples": len(training_rows),
                            "positive_examples": sum(row["interview"] for row in training_rows),
                            "negative_examples": sum(not row["interview"] for row in training_rows),
                            "weave_reference": training_url,
                            "characteristics_source": characteristics_artifact_path,
                            "split_ratio": split_ratio
                        }
                        
                        # Create and log evaluation artifact
                        evaluation_artifact = wandb.Artifact(
                            name="evaluation_dataset",
                            type="dataset",
                            description="Generated applications for evaluation"
                        )
                        evaluation_artifact.add(evaluation_table, "evaluation_dataset")
                        
                        # Add metadata to evaluation artifact
                        evaluation_artifact.metadata = {
                            "r_score": r_score,
                            "num_examples": len(evaluation_rows),
                            "positive_examples": sum(row["interview"] for row in evaluation_rows),
                            "negative_examples": sum(not row["interview"] for row in evaluation_rows),
                            "weave_reference": evaluation_url,
                            "characteristics_source": characteristics_artifact_path,
                            "split_ratio": 1 - split_ratio
                        }
                        
                        # Log artifacts
                        run.log_artifact(training_artifact)
                        run.log_artifact(evaluation_artifact)
                        
                        # Finish the run
                        wandb.finish()
                        
                        # Display success message with links
                        st.success(f"Datasets created and published successfully!")
                        
                        # Create columns for displaying dataset info
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Fine-tuning Dataset")
                            st.markdown(f"[View Weave dataset]({training_url})")
                            st.markdown(f"[View W&B artifact](https://wandb.ai/{run.entity}/{run.project}/artifacts/{training_artifact.type}/{training_artifact.name}/v0)")
                            st.metric("Examples", len(training_rows))
                            st.metric("Positive Examples", sum(row["interview"] for row in training_rows))
                            st.metric("Negative Examples", sum(not row["interview"] for row in training_rows))
                        
                        with col2:
                            st.subheader("Evaluation Dataset")
                            st.markdown(f"[View Weave dataset]({evaluation_url})")
                            st.markdown(f"[View W&B artifact](https://wandb.ai/{run.entity}/{run.project}/artifacts/{evaluation_artifact.type}/{evaluation_artifact.name}/v0)")
                            st.metric("Examples", len(evaluation_rows))
                            st.metric("Positive Examples", sum(row["interview"] for row in evaluation_rows))
                            st.metric("Negative Examples", sum(not row["interview"] for row in evaluation_rows))
                        
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