import unittest
from data.loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def test_dataset_splits_non_overlapping(self):
        loader = DataLoader("ag_news")
        splits = loader.load_dataset_splits(
            seed=42,
            train_initial_size=100,
            al_pool_size=100,
            val_calib_size=50,
            test_size=50
        )

        df_train_initial = splits["train_initial"]
        df_al_pool = splits["al_pool"]
        df_val_calib = splits["val_calibration"]
        df_test = splits["test"]

        self.assertEqual(len(df_train_initial), 100)
        self.assertEqual(len(df_al_pool), 100)
        self.assertEqual(len(df_val_calib), 50)
        self.assertEqual(len(df_test), 50)

        self.assertFalse(df_train_initial["text"].isnull().any())
        self.assertFalse(df_test["text"].isnull().any())

if __name__ == "__main__":
    unittest.main()
