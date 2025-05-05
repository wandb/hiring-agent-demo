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
    reason_comp_prompt: weave.StringPrompt

    @weave.op
    def score(self, reason: str, model_output: dict) -> dict:
        model = ChatOpenAI(
            model=self.model_id,
            max_retries=5,             # Built-in retry mechanism
            request_timeout=60.0,      # Longer timeout
            response_format={"type": "json"}).with_structured_output(ReasonComparison)
        
        prompt = self.reason_comp_prompt.format(p1_reasoning=reason, p2_reasoning=model_output["reason"])
        evaluation = model.invoke(prompt)
        
        # Convert to dictionary format for consistent handling
        return {
            'role_domain_fit': evaluation.role_domain_fit,
            'technical_rigor': evaluation.technical_rigor,
            'values_alignment': evaluation.values_alignment,
            'collaboration_style': evaluation.collaboration_style,
            'communication_clarity': evaluation.communication_clarity,
            'fairness_objectivity': evaluation.fairness_objectivity,
            'impact_orientation': evaluation.impact_orientation,
            'decision_consistency': evaluation.decision_consistency,
            'pass_fail': evaluation.pass_fail,
            'rationale': evaluation.rationale,
            # Add flattened score fields for easier summarization
            'role_domain_fit_score': evaluation.role_domain_fit.score,
            'technical_rigor_score': evaluation.technical_rigor.score,
            'values_alignment_score': evaluation.values_alignment.score,
            'collaboration_style_score': evaluation.collaboration_style.score,
            'communication_clarity_score': evaluation.communication_clarity.score,
            'fairness_objectivity_score': evaluation.fairness_objectivity.score,
            'impact_orientation_score': evaluation.impact_orientation.score,
            'decision_consistency_score': evaluation.decision_consistency.score
        }
        
    def summarize(self, score_rows: list) -> dict:
        """Aggregate reason match scores across all examples.
        
        Args:
            score_rows: List of score dictionaries from all examples
            
        Returns:
            Dictionary with aggregated reason matching metrics
        """
        # Initialize metric counters
        metrics = [
            'role_domain_fit', 'technical_rigor', 'values_alignment', 
            'collaboration_style', 'communication_clarity', 'fairness_objectivity',
            'impact_orientation', 'decision_consistency'
        ]
        
        # Initialize counters for each metric
        metric_sums = {metric: 0 for metric in metrics}
        metric_counts = {metric: 0 for metric in metrics}
        
        # Count passes
        pass_count = 0
        total_examples = len(score_rows)
        
        # Extract scores directly from score rows
        for row in score_rows:
            # Count passes based on pass_fail field
            if 'pass_fail' in row and row['pass_fail'].lower() == 'pass':
                pass_count += 1
            
            # Extract scores from flattened fields
            for metric in metrics:
                score_key = f"{metric}_score"
                if score_key in row:
                    try:
                        score = int(row[score_key])
                        metric_sums[metric] += score
                        metric_counts[metric] += 1
                    except (ValueError, TypeError):
                        # Skip invalid scores
                        pass
        
        # Calculate average scores for each metric
        avg_scores = {}
        for metric in metrics:
            avg_scores[f'avg_{metric}_score'] = (
                metric_sums[metric] / metric_counts[metric] 
                if metric_counts[metric] > 0 else 0
            )
        
        # Calculate pass rate
        pass_rate = pass_count / total_examples if total_examples > 0 else 0
        
        # Calculate overall average score across all metrics
        total_score_sum = sum(metric_sums.values())
        total_score_count = sum(metric_counts.values())
        overall_avg_score = total_score_sum / total_score_count if total_score_count > 0 else 0
        
        return {
            **avg_scores,
            'pass_rate': pass_rate,
            'pass_count': pass_count,
            'overall_avg_score': overall_avg_score,
            'total_examples': total_examples
        }