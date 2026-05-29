from datasets import load_dataset
import pandas as pd

class DataLoader:
    def __init__(self, dataset_name: str = "ag_news"):
        self.dataset_name = dataset_name
        self._categories = None

    def load_data(self, split: str = "test", num_samples: int = 100):
        dataset = load_dataset(self.dataset_name, split=split)
        
        # Select a subset
        subset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
        
        # Automatically detect the text column
        text_col = next((col for col in dataset.column_names if col in ["text", "content", "sentence", "document", "paragraph"]), None)
        if not text_col:
            text_col = dataset.column_names[0]
            
        # Automatically extract categories from features
        features = dataset.features
        mapping = {}
        if "label" in features and hasattr(features["label"], "names"):
            categories = features["label"].names
            mapping = {i: name for i, name in enumerate(categories)}
            self._categories = categories
        else:
            unique_labels = sorted(list(set(dataset["label"])))
            categories = [f"Class_{lbl}" for lbl in unique_labels]
            mapping = {lbl: name for lbl, name in zip(unique_labels, categories)}
            self._categories = categories
            
        data = []
        for item in subset:
            text_val = item.get(text_col, "")
            label_id = item.get("label", -1)
            data.append({
                "text": str(text_val),
                "label": mapping.get(label_id, "Unknown")
            })
            
        return pd.DataFrame(data)

    def get_categories(self):
        if self._categories is None:
            # Force load a tiny subset to extract categories
            self.load_data(split="test", num_samples=1)
        return self._categories

    def generate_synthetic_samples(self, categories: list, n: int = 5):
        """
        Generates "Virtual Leaf" samples for cold-start.
        """
        synthetic = []
        for cat in categories:
            for i in range(n):
                synthetic.append({
                    "text": f"This is a synthetic sample for category {cat}. It contains keywords related to {cat}.",
                    "label": cat
                })
        return pd.DataFrame(synthetic)
