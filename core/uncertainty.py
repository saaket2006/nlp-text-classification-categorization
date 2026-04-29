import numpy as np
from scipy.stats import entropy

class UncertaintyEngine:
    def __init__(self, tau_1: float, tau_2: float):
        """
        tau_1: Entropy threshold (escalate if > tau_1)
        tau_2: Confidence threshold (escalate if Max_Prob < tau_2)
        """
        self.tau_1 = tau_1
        self.tau_2 = tau_2

    def calculate_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculates Shannon Entropy for the given probability distribution.
        """
        # Normalize if not already
        probs = probabilities / np.sum(probabilities)
        return float(entropy(probs, base=2))

    def should_escalate(self, probabilities: np.ndarray) -> bool:
        """
        Escalation logic: Entropy > tau_1 OR Max_Prob < tau_2
        """
        ent = self.calculate_entropy(probabilities)
        max_prob = np.max(probabilities)
        
        if ent > self.tau_1 or max_prob < self.tau_2:
            return True
        return False

    def get_metrics(self, probabilities: np.ndarray):
        """
        Returns entropy and confidence.
        """
        ent = self.calculate_entropy(probabilities)
        conf = float(np.max(probabilities))
        return ent, conf
