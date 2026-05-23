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
                    # Selective T2 Override: Trust T2 when T1 uncertain, T1 entropy is low enough, and T2 is unanimous
                    t2_vote_agreement = t2_res.get("vote_agreement", 0.0)
                    if t1_conf < 0.80 and t1_entropy < 0.95 and t2_vote_agreement >= 1.0:
                        # T1 was unsure AND all T2 votes agree — high confidence correction
                        output["tier"] = 2
                        output["final_label"] = t2_labels
                        output["rationale"] += f" | T2 Override (T1 conf {t1_conf:.2f}, T2 unanimous). Trusting T2."
                    else:
                        # Fall back to T1 to optimize cost and avoid unnecessary human escalation
                        output["tier"] = 2
                        output["final_label"] = [t1_label]
                        output["rationale"] += f" | T1/T2 Disagreement. Confidence bounds not met. Falling back to T1."
            else:
                # LLM Failure -> Human
                output["tier"] = 3
                output["rationale"] = "Tier 2 failed. Escalating to Human."
                
        return output
