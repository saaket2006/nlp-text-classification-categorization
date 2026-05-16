from models.tier1_model import Tier1Model
from models.tier2_llm import Tier2LLM
from core.uncertainty import UncertaintyEngine
import numpy as np

class TieredRouter:
    def __init__(self, tier1, tier2, uncertainty_engine, categories, entropy_threshold, conf_threshold, extreme_entropy_cap=1.2):
        self.tier1 = tier1
        self.tier2 = tier2
        self.llm = tier2
        self.uncertainty_engine = uncertainty_engine
        self.categories = categories
        self.entropy_threshold = entropy_threshold
        self.conf_threshold = conf_threshold
        self.extreme_entropy_cap = extreme_entropy_cap
        self.id_to_cat = {i: cat for i, cat in enumerate(categories)}
        self.cat_to_id = {cat: i for i, cat in enumerate(categories)}

    def process_sample(self, text: str) -> dict:
        t1_idx, t1_probs, t1_conf = self.tier1.predict(text)
        t1_label = self.id_to_cat[t1_idx]
        t1_entropy = self.uncertainty_engine.calculate_entropy(t1_probs)
        
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

        # 1. Absolute Hard Stop: Extreme Uncertainty -> Human
        if t1_entropy > self.extreme_entropy_cap:
            output["tier"] = 3
            output["rationale"] = f"Extreme T1 uncertainty ({t1_entropy:.2f} > {self.extreme_entropy_cap}). Direct Human Escalation."
            return output

        # 2. Uncertainty Window: Potential Tier 2 / Tier 3
        if t1_entropy > self.entropy_threshold or t1_conf < self.conf_threshold:
            # Escalate to Tier 2
            t2_res, raw_res = self.llm.predict(text, self.categories, t1_label=t1_label)
            
            if t2_res:
                t2_labels = t2_res.get("labels", [])
                t2_conf = t2_res.get("confidence", 0.5)
                output["rationale"] = f"T1 uncertain. Tier 2 suggests {t2_labels} (conf: {t2_conf})."
                
                if t1_label in t2_labels:
                    # Consensus - stay with T1 but mark as Tier 2 processed
                    output["tier"] = 2
                    output["rationale"] += " | T1/T2 Consensus reached."
                else:
                    # Disagreement - be very careful
                    if (t2_conf >= 1.01):
                        # Free Gain path: (Disabled to ensure stability)
                        output["tier"] = 2
                        output["final_label"] = t2_labels
                        output["rationale"] += " | (Override Disabled for Stability)."
                    else:
                        # Standard disagreement -> Human
                        output["tier"] = 3
                        output["disagreement"] = True
                        output["rationale"] += " | T1/T2 Disagreement. Escalating to Human for safety."
            else:
                # LLM Failure -> Human
                output["tier"] = 3
                output["rationale"] = "Tier 2 failed. Escalating to Human."
                
        return output
