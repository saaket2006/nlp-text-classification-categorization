from models.tier1_model import Tier1Model
from models.tier2_llm import Tier2LLM
from core.uncertainty import UncertaintyEngine
import numpy as np

class TieredRouter:
    def __init__(self, tier1: Tier1Model, tier2: Tier2LLM, uncertainty_engine: UncertaintyEngine, categories: list):
        self.tier1 = tier1
        self.tier2 = tier2
        self.ue = uncertainty_engine
        self.categories = categories
        self.id_to_cat = {i: cat for i, cat in enumerate(categories)}
        self.cat_to_id = {cat: i for i, cat in enumerate(categories)}

    def process_sample(self, text: str):
        # Tier 1 Inference
        t1_idx, t1_probs, t1_conf = self.tier1.predict(text)
        t1_label = self.id_to_cat[t1_idx]
        t1_entropy = self.ue.calculate_entropy(t1_probs)
        
        output = {
            "text": text,
            "tier": 1,
            "predicted_labels": [t1_label],
            "confidence": float(t1_conf),
            "entropy": float(t1_entropy),
            "disagreement": False,
            "final_label": [t1_label],
            "rationale": "Tier 1 classified with high confidence."
        }
        
        # Step 1: Uncertainty Escalation
        if self.ue.should_escalate(t1_probs):
            output["tier"] = 2
            t2_result, raw_response = self.tier2.predict(text, self.categories)
            
            if t2_result:
                t2_labels = t2_result.get("labels", [])
                t2_conf = t2_result.get("confidence", 0.0)
                t2_reasoning = t2_result.get("reasoning", "")
                
                output["predicted_labels"] = t2_labels
                output["confidence"] = float(t2_conf)
                output["rationale"] = t2_reasoning
                output["final_label"] = t2_labels
                
                # Step 2: Disagreement Logic
                # If Tier 2 disagrees with Tier 1 (if Tier 1 had a clear winner but was uncertain)
                # or if Tier 2 labels are not in categories
                if t1_label not in t2_labels:
                    output["disagreement"] = True
                    output["tier"] = 3
                    output["rationale"] += " | Disagreement between Tier 1 and Tier 2. Escalating to Tier 3."
            else:
                # Invalid Tier 2 response
                output["tier"] = 3
                output["rationale"] = "Tier 2 response invalid. Escalating to Tier 3."
                
        # Final validation (Tier 3 Simulation if needed)
        # In a real scenario, this waits for human input.
        if output["tier"] == 3:
            # For automation, we might flag this for human or use a fallback 'GROUND_TRUTH'
            pass

        return output
