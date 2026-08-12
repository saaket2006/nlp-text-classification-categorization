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

    def load_dataset_splits(self, seed: int = 42, train_initial_size: int = 1500, al_pool_size: int = 1000, val_calib_size: int = 500, test_size: int = 500):
        """
        Loads non-overlapping data partitions to guarantee 0 test-set leakage:
        - train_initial: Initial labeled set for Tier 1 pretraining
        - al_pool: Pool of unlabeled samples for active learning escalation
        - val_calibration: Validation set reserved for threshold tuning and cap calibration
        - test: Official test set reserved STRICTLY for final evaluation (never touched during training or tuning)
        """
        raw_train = load_dataset(self.dataset_name, split="train")
        raw_test = load_dataset(self.dataset_name, split="test")

        # Automatically detect text column
        text_col = next((col for col in raw_train.column_names if col in ["text", "content", "sentence", "document", "paragraph"]), None)
        if not text_col:
            text_col = raw_train.column_names[0]

        # Extract categories
        features = raw_train.features
        mapping = {}
        if "label" in features and hasattr(features["label"], "names"):
            categories = features["label"].names
            mapping = {i: name for i, name in enumerate(categories)}
            self._categories = categories
        else:
            unique_labels = sorted(list(set(raw_train["label"])))
            categories = [f"Class_{lbl}" for lbl in unique_labels]
            mapping = {lbl: name for lbl, name in zip(unique_labels, categories)}
            self._categories = categories

        # Shuffle raw splits with seed
        shuffled_train = raw_train.shuffle(seed=seed)
        shuffled_test = raw_test.shuffle(seed=seed)

        # Slice train splits
        idx0 = 0
        idx1 = min(train_initial_size, len(shuffled_train))
        idx2 = min(idx1 + al_pool_size, len(shuffled_train))
        idx3 = min(idx2 + val_calib_size, len(shuffled_train))

        ds_train_initial = shuffled_train.select(range(idx0, idx1))
        ds_al_pool = shuffled_train.select(range(idx1, idx2))
        ds_val_calib = shuffled_train.select(range(idx2, idx3))
        ds_test = shuffled_test.select(range(min(test_size, len(shuffled_test))))

        def to_df(ds):
            data = []
            for item in ds:
                data.append({
                    "text": str(item.get(text_col, "")),
                    "label": mapping.get(item.get("label", -1), "Unknown")
                })
            return pd.DataFrame(data)

        return {
            "train_initial": to_df(ds_train_initial),
            "al_pool": to_df(ds_al_pool),
            "val_calibration": to_df(ds_val_calib),
            "test": to_df(ds_test)
        }

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
