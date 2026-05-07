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
            "t1_label": t1_label,
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
                t2_conf = float(t2_result.get("confidence", 0.0))
                t2_reasoning = t2_result.get("reasoning", "")
                
                # Only trust Tier 2 if it's highly confident (> 0.8)
                if t2_conf > 0.8:
                    output["predicted_labels"] = t2_labels
                    output["confidence"] = t2_conf
                    output["rationale"] = t2_reasoning
                    output["final_label"] = t2_labels
                    
                    # Step 2: Disagreement Logic
                    if t1_label not in t2_labels:
                        if t2_conf > 0.9:
                            output["disagreement"] = False
                            output["rationale"] += " | Strong Tier 2 confidence overrides Tier 1 disagreement."
                        else:
                            output["disagreement"] = True
                            output["tier"] = 3
                            output["rationale"] += " | Disagreement with moderate confidence. Escalating to Human."
                else:
                    # Tier 2 is unsure, stay with Tier 1
                    output["rationale"] = f"Tier 2 uncertain ({t2_conf}). Retaining Tier 1 prediction."
                    output["tier"] = 2 # Mark as T2 used but rejected
            else:
                # Invalid Tier 2 response
                output["tier"] = 3
                output["rationale"] = "Tier 2 response invalid. Escalating to Tier 3."
                
        # Final Anchor: Extreme Uncertainty always goes to Human
        if t1_entropy > 1.2:
            output["tier"] = 3
            output["rationale"] = "Extreme uncertainty detected. Mandatory human review."
                
        # Final validation (Tier 3 Simulation if needed)
        # In a real scenario, this waits for human input.
        if output["tier"] == 3:
            # For automation, we might flag this for human or use a fallback 'GROUND_TRUTH'
            pass

        return output
