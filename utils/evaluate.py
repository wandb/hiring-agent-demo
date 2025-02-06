import weave
from weave import Scorer
from typing import List
from langchain_openai import ChatOpenAI

from utils.prompt import EvaluationDataset, EvaluationExample, ReasonComparison, reason_comp_prompt
from utils.generate import generate_application, save_as_pdf
from utils.prepro import extract_text_from_pdf

# tp score whether the interview decision match
@weave.op()
def decision_match(interview: bool, model_output: dict) -> dict:
    return {'decision_match': interview == model_output['interview']}

# to score whether the reasoning matches
class ReasonScorer(Scorer):
    model_id: str = "gpt-4o"

    @weave.op
    def score(self, reason: str, model_output: dict) -> dict:
        model = ChatOpenAI(
            model=self.model_id, 
            response_format={"type": "json"}).with_structured_output(ReasonComparison)
        prompt = reason_comp_prompt.format(p1_reasoning=reason, p2_reasoning=model_output["reason"])
        reason_match = model.invoke(prompt)
        return {'reason_match': reason_match}

@weave.op()
def generate_dataset(offer_list: List[str], generation_model: str, num_app: int) -> EvaluationDataset:    
    examples = []
    for offer_path in offer_list:
        # Extract text from offer PDF
        offer_text = extract_text_from_pdf(offer_path)
        offer_id = offer_path.split("/")[-1].split(".")[0]
        print("offer_id: ", offer_id)
        # Generate positive and negative applications
        for i in range(num_app):
            # Generate positive application
            pos_app_content, pos_reasoning = generate_application(
                generation_model=generation_model, job_offer=offer_text, positive=True) 
            pos_app_path = f"utils/data/applications/sample_application_pos_{i}_{offer_id}.pdf"
            save_as_pdf(pos_app_content, pos_app_path)
            
            # Generate negative application (with different qualifications)
            neg_app_content, neg_reasoning = generate_application(
                generation_model=generation_model, job_offer=offer_text, positive=False)
            neg_app_path = f"utils/data/applications/sample_application_neg_{i}_{offer_id}.pdf"
            save_as_pdf(neg_app_content, neg_app_path)
            
            # Add positive example
            examples.append(EvaluationExample(
                offer_pdf=offer_path,
                offer_text=offer_text,
                application_pdf=pos_app_path,
                application_text=pos_app_content,
                interview=True,
                reason=pos_reasoning
            ))
            
            # Add negative example  
            examples.append(EvaluationExample(
                offer_pdf=offer_path,
                offer_text=offer_text,
                application_pdf=neg_app_path,
                application_text=neg_app_content,
                interview=False,
                reason=neg_reasoning
            ))
            
    return EvaluationDataset(examples=examples)