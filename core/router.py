from models.tier1_model import Tier1Model
from models.tier2_llm import Tier2LLM
from core.uncertainty import UncertaintyEngine
import numpy as np
import random

class TieredRouter:
    def __init__(self, tier1, tier2, uncertainty_engine, categories, entropy_threshold, conf_threshold, extreme_entropy_cap=1.2, dataset_name: str = None, mode: str = "STANDARD", budget_ratios: tuple = (0.60, 0.30, 0.10), human_error_rate: float = 0.05):
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
        self.dataset_name = dataset_name
        self.mode = mode.upper()
        self.budget_ratios = budget_ratios  # (p1, p2, p3) for BUDGET_MATCHED_RANDOM mode
        self.human_error_rate = human_error_rate
        
        # Thresholds for AL Baselines (calibrated dynamically to match escalation rates)
        self.al_budget_prob = 0.10
        self.al_least_conf_threshold = 0.5
        self.al_entropy_threshold = 0.5
        self.al_margin_threshold = 0.1

    def process_sample(self, text: str, sample_idx: int = None) -> dict:
        import time
        start_time = time.perf_counter()
        output = self._process_sample_core(text, sample_idx)
        output["latency"] = time.perf_counter() - start_time
        
        # Construct final probability vector for calibration and Brier score metrics
        t1_probs = output.get("t1_probs", [])
        tier = output["tier"]
        final_label = output["final_label"][0] if output["final_label"] else None
        
        if tier == 1:
            output["final_probs"] = t1_probs
        elif tier == 2:
            c = output["confidence"]
            K = len(self.categories)
            probs = [(1.0 - c) / (K - 1) if K > 1 else 0.0] * K
            if final_label in self.cat_to_id:
                probs[self.cat_to_id[final_label]] = c
            output["final_probs"] = probs
        elif tier == 3:
            # Calibrate Tier 3 probability vector based on true human oracle correctness rate (1.0 - human_error_rate)
            c = 1.0 - self.human_error_rate
            K = len(self.categories)
            probs = [(1.0 - c) / (K - 1) if K > 1 else 0.0] * K
            if final_label in self.cat_to_id:
                probs[self.cat_to_id[final_label]] = c
            else:
                probs[0] = c
            output["final_probs"] = probs
            
        return output

    def _process_sample_core(self, text: str, sample_idx: int = None) -> dict:
        t1_idx, t1_probs, t1_conf = self.tier1.predict(text)
        t1_label = self.id_to_cat[t1_idx]
        t1_entropy = self.uncertainty_engine.calculate_entropy(t1_probs)

        output = {
            "text": text,
            "tier": 1,
            "t1_label": t1_label,
            "predicted_labels": [t1_label],
            "t1_confidence": float(t1_conf),
            "confidence": float(t1_conf),  # Final system confidence
            "entropy": float(t1_entropy),
            "disagreement": False,
            "final_label": [t1_label],
            "rationale": "Tier 1 classified with high confidence.",
            "t1_probs": [float(p) for p in t1_probs]
        }

        # --- MODE 1: BUDGET_MATCHED_RANDOM ---
        if self.mode == "BUDGET_MATCHED_RANDOM":
            p1, p2, p3 = self.budget_ratios
            r = random.random()
            if r < p1:
                output["tier"] = 1
                output["rationale"] = "Budget-matched random routing: Tier 1"
            elif r < p1 + p2:
                output["tier"] = 2
                t2_res, _ = self.llm.predict(text, self.categories, t1_label=t1_label, sample_idx=sample_idx)
                if t2_res:
                    t2_labels = t2_res.get("labels", [])
                    t2_conf = t2_res.get("confidence", 0.5)
                    output["confidence"] = float(t2_conf)
                    output["final_label"] = t2_labels if t2_labels else [t1_label]
                    output["rationale"] = f"Budget-matched random routing: Tier 2 suggests {t2_labels} (conf: {t2_conf})."
                else:
                    output["tier"] = 3
                    output["confidence"] = 1.0 - self.human_error_rate
                    output["rationale"] = "Tier 2 failed in random routing. Escalating to Tier 3."
            else:
                output["tier"] = 3
                output["confidence"] = 1.0 - self.human_error_rate
                output["rationale"] = "Budget-matched random routing: Tier 3 (Direct Human Escalation)"
            return output

        # --- MODE 2: TIER2_ONLY ---
        if self.mode == "TIER2_ONLY":
            t2_res, _ = self.llm.predict(text, self.categories, t1_label=t1_label, sample_idx=sample_idx)
            output["tier"] = 2
            if t2_res:
                t2_labels = t2_res.get("labels", [])
                t2_conf = t2_res.get("confidence", 0.5)
                output["confidence"] = float(t2_conf)
                output["final_label"] = t2_labels if t2_labels else [t1_label]
                output["rationale"] = f"Tier 2 Only Baseline suggests {t2_labels} (conf: {t2_conf})."
            else:
                output["tier"] = 3
                output["confidence"] = 1.0 - self.human_error_rate
                output["rationale"] = "Tier 2 failed in baseline. Escalating to Tier 3."
            return output

        # --- MODE 3: NO_TIER2 ---
        if self.mode == "NO_TIER2":
            if t1_entropy > self.extreme_entropy_cap or t1_entropy > self.entropy_threshold or t1_conf < self.conf_threshold:
                output["tier"] = 3
                output["confidence"] = 1.0 - self.human_error_rate
                output["rationale"] = "T1 uncertain. Bypassing Tier 2, escalating directly to Tier 3."
            return output

        # --- MODE 4: NO_ENTROPY ---
        if self.mode == "NO_ENTROPY":
            if t1_conf < self.conf_threshold:
                t2_res, _ = self.llm.predict(text, self.categories, t1_label=t1_label, sample_idx=sample_idx)
                if t2_res:
                    t2_labels = t2_res.get("labels", [])
                    t2_conf = t2_res.get("confidence", 0.5)
                    output["tier"] = 2
                    output["confidence"] = float(t2_conf)
                    output["final_label"] = t2_labels if t2_labels else [t1_label]
                    output["rationale"] = f"No Entropy Routing: T1 low confidence ({t1_conf:.2f}). Tier 2 suggests {t2_labels}."
                else:
                    output["tier"] = 3
                    output["confidence"] = 1.0 - self.human_error_rate
                    output["rationale"] = "Tier 2 failed. Escalating to Tier 3."
            return output

        # --- ACTIVE LEARNING BASELINE MODES ---
        if self.mode == "AL_RANDOM":
            r = random.random()
            if r < self.al_budget_prob:
                output["tier"] = 3
                output["confidence"] = 1.0 - self.human_error_rate
                output["rationale"] = "Active Learning Random Selection Baseline (Tier 3)"
            else:
                output["tier"] = 1
                output["rationale"] = "Active Learning Random Selection Baseline (Tier 1)"
            return output

        if self.mode == "AL_LEAST_CONFIDENCE":
            if t1_conf < self.al_least_conf_threshold:
                output["tier"] = 3
                output["confidence"] = 1.0 - self.human_error_rate
                output["rationale"] = f"Least Confidence AL Baseline: T1 confidence {t1_conf:.2f} below threshold."
            else:
                output["tier"] = 1
                output["rationale"] = f"Least Confidence AL Baseline: T1 confidence {t1_conf:.2f} above threshold."
            return output

        if self.mode == "AL_ENTROPY":
            if t1_entropy > self.al_entropy_threshold:
                output["tier"] = 3
                output["confidence"] = 1.0 - self.human_error_rate
                output["rationale"] = f"Entropy AL Baseline: T1 entropy {t1_entropy:.2f} above threshold."
            else:
                output["tier"] = 1
                output["rationale"] = f"Entropy AL Baseline: T1 entropy {t1_entropy:.2f} below threshold."
            return output

        if self.mode == "AL_MARGIN":
            sorted_probs = sorted(t1_probs, reverse=True)
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
            output["margin"] = float(margin)
            if margin < self.al_margin_threshold:
                output["tier"] = 3
                output["confidence"] = 1.0 - self.human_error_rate
                output["rationale"] = f"Margin AL Baseline: T1 margin {margin:.2f} below threshold."
            else:
                output["tier"] = 1
                output["rationale"] = f"Margin AL Baseline: T1 margin {margin:.2f} above threshold."
            return output

        # --- MODE 5: STANDARD (Tri-Tiered Active Learning Router) ---
        # 1. Absolute Hard Stop: Extreme Uncertainty -> Human
        if t1_entropy > self.extreme_entropy_cap:
            output["tier"] = 3
            output["confidence"] = 1.0 - self.human_error_rate
            output["rationale"] = f"Extreme T1 uncertainty ({t1_entropy:.2f} > {self.extreme_entropy_cap}). Direct Human Escalation."
            return output

        # 2. Uncertainty Window: Potential Tier 2 / Tier 3
        if t1_entropy > self.entropy_threshold or t1_conf < self.conf_threshold:
            # Escalate to Tier 2
            t2_res, raw_res = self.llm.predict(text, self.categories, t1_label=t1_label, sample_idx=sample_idx)

            if t2_res:
                t2_labels = t2_res.get("labels", [])
                t2_conf = t2_res.get("confidence", 0.5)
                t2_label = t2_labels[0] if t2_labels else None
                output["rationale"] = f"T1 uncertain. Tier 2 suggests {t2_labels} (conf: {t2_conf})."

                if t1_label in t2_labels:
                    # Consensus - stay with T1 but mark as Tier 2 processed
                    output["tier"] = 2
                    output["confidence"] = float(max(t1_conf, t2_conf))
                    output["rationale"] += " | T1/T2 Consensus reached."
                else:
                    # Selective T2 Override
                    t2_vote_agreement = t2_res.get("vote_agreement", 0.0)
                    num_classes = len(self.categories)
                    max_entropy = np.log2(num_classes) if num_classes > 0 else 1.0

                    conf_floor = 0.70 if num_classes > 5 else 0.85
                    entropy_ceiling = 0.75 * max_entropy

                    # Custom override rules for DBpedia
                    if self.dataset_name == "dbpedia_14" and t2_label is not None:
                        is_short = len(text) < 50
                        is_educational = t2_label == "EducationalInstitution"

                        is_org_to_person = (
                            t1_label in ["Company", "EducationalInstitution", "Building", "WrittenWork"] and 
                            t2_label in ["OfficeHolder", "Artist", "Athlete"]
                        )

                        is_film_confusion = False
                        if t1_label == "Film" and t2_label in ["Album", "WrittenWork"]:
                            prefix_lower = text[:50].lower()
                            allowed_written_words = ["play", "book", "novel", "magazine", "journal", "written", "published", "author"]
                            if not any(w in prefix_lower for w in allowed_written_words):
                                is_film_confusion = True

                        should_override = (
                            t1_conf < 0.96 and 
                            t2_vote_agreement >= 0.6 and 
                            not is_short and 
                            not is_educational and
                            not is_org_to_person and
                            not is_film_confusion
                        )
                    else:
                        if num_classes == 2:
                            should_override = (t1_conf < conf_floor and t2_vote_agreement >= 0.6)
                        else:
                            should_override = (t1_conf < conf_floor and t1_entropy < entropy_ceiling and t2_vote_agreement >= 0.6)

                    override_reason = f"T2 Override (T1 conf {t1_conf:.2f}, T2 agreement {t2_vote_agreement:.2f}). Trusting T2."

                    if should_override:
                        output["tier"] = 2
                        output["final_label"] = t2_labels
                        output["confidence"] = float(t2_conf)
                        output["rationale"] += f" | {override_reason}"
                    else:
                        # Fall back to T1
                        output["tier"] = 2
                        output["final_label"] = [t1_label]
                        output["confidence"] = float(t1_conf)
                        output["rationale"] += f" | T1/T2 Disagreement. Confidence bounds not met. Falling back to T1."
            else:
                # LLM Failure -> Human
                output["tier"] = 3
                output["confidence"] = 1.0 - self.human_error_rate
                output["rationale"] = "Tier 2 failed. Escalating to Human."

        return output
