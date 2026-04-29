from datasets import load_dataset
import pandas as pd

class DataLoader:
    def __init__(self, dataset_name: str = "ag_news"):
        self.dataset_name = dataset_name

    def load_data(self, split: str = "test", num_samples: int = 100):
        dataset = load_dataset(self.dataset_name, split=split)
        # Select a subset
        subset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
        
        # AG News Categories: 0: World, 1: Sports, 2: Business, 3: Sci/Tech
        mapping = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        
        data = []
        for item in subset:
            data.append({
                "text": item["text"],
                "label": mapping.get(item["label"], "Unknown")
            })
            
        return pd.DataFrame(data)

    def generate_synthetic_samples(self, categories: list, n: int = 5):
        """
        Generates "Virtual Leaf" samples for cold-start (placeholder).
        """
        synthetic = []
        for cat in categories:
            for i in range(n):
                synthetic.append({
                    "text": f"This is a synthetic sample for category {cat}. It contains keywords related to {cat}.",
                    "label": cat
                })
        return pd.DataFrame(synthetic)
