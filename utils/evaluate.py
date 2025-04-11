import weave
from weave import Scorer
from langchain_openai import ChatOpenAI

from utils.prompt import ReasonComparison, reason_comp_prompt

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
            max_retries=5,             # Built-in retry mechanism
            request_timeout=60.0,      # Longer timeout
            response_format={"type": "json"}).with_structured_output(ReasonComparison)
        prompt = reason_comp_prompt.format(p1_reasoning=reason, p2_reasoning=model_output["reason"])
        reason_match = model.invoke(prompt)
        return {'reason_match': reason_match}