import weave
from weave import Scorer
from langchain_openai import ChatOpenAI

from utils.prompt import ReasonComparison, reason_comp_prompt

# to score whether the interview decision match
class DecisionScorer(Scorer):
    model_id: str = "gpt-4o"  # Default model ID, though not used directly in this scorer
    
    @weave.op
    def score(self, interview: bool, model_output: dict) -> dict:
        """Score whether the model's interview decision matches the ground truth.
        
        Args:
            interview: The ground truth interview decision (True/False)
            model_output: The model's output containing the interview decision
        """
        # Get the model's predicted interview decision
        model_interview = model_output['interview']
        
        # Compute classification metrics components
        tp = 1 if interview and model_interview else 0
        fp = 1 if not interview and model_interview else 0
        fn = 1 if interview and not model_interview else 0
        tn = 1 if not interview and not model_interview else 0
        
        # Calculate precision, recall, and F1 for this single example
        # (these will be aggregated across examples later)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = 1 if interview == model_interview else 0
        
        return {
            'decision_match': interview == model_interview,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
        
    def summarize(self, score_rows: list) -> dict:
        """Aggregate metrics across all examples to compute overall precision, recall, and F1.
        
        Args:
            score_rows: List of score dictionaries from all examples
            
        Returns:
            Dictionary with aggregated metrics
        """
        # Sum up TP, FP, FN, TN across all examples
        total_tp = sum(row['tp'] for row in score_rows)
        total_fp = sum(row['fp'] for row in score_rows)
        total_fn = sum(row['fn'] for row in score_rows)
        total_tn = sum(row['tn'] for row in score_rows)
        
        # Calculate overall metrics from the aggregated counts
        accuracy = (total_tp + total_tn) / len(score_rows) if len(score_rows) > 0 else 0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Count matches
        matches = sum(1 for row in score_rows if row['decision_match'])
        match_rate = matches / len(score_rows) if len(score_rows) > 0 else 0
        
        return {
            'decision_match_rate': match_rate,
            'total_examples': len(score_rows),
            'total_matches': matches,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_tp': total_tp,
            'total_fp': total_fp, 
            'total_fn': total_fn,
            'total_tn': total_tn
        }

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
        
    def summarize(self, score_rows: list) -> dict:
        """Aggregate reason match scores across all examples.
        
        Args:
            score_rows: List of score dictionaries from all examples
            
        Returns:
            Dictionary with aggregated reason matching metrics
        """
        # Extract match scores (assuming reason_match contains a score field)
        # Note: This depends on the structure of the ReasonComparison class
        match_scores = []
        for row in score_rows:
            if isinstance(row['reason_match'], dict) and 'score' in row['reason_match']:
                match_scores.append(row['reason_match']['score'])
            elif hasattr(row['reason_match'], 'score'):
                match_scores.append(row['reason_match'].score)
            else:
                # If score is not available, try to use a default value or skip
                match_scores.append(0)
                
        # Calculate aggregate statistics
        avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0
        max_score = max(match_scores) if match_scores else 0
        min_score = min(match_scores) if match_scores else 0
        
        # Count perfect matches (assuming score of 1.0 means perfect match)
        perfect_matches = sum(1 for score in match_scores if score >= 0.9)
        perfect_match_rate = perfect_matches / len(match_scores) if match_scores else 0
        
        return {
            'avg_reason_match_score': avg_match_score,
            'max_reason_match_score': max_score,
            'min_reason_match_score': min_score,
            'perfect_matches': perfect_matches,
            'perfect_match_rate': perfect_match_rate,
            'total_examples': len(score_rows)
        }