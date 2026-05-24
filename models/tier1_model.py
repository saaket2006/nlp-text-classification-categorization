import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Tier1Model:
    def __init__(self, model_name: str, num_labels: int, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)

    def predict(self, text: str):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        
        return predicted_idx, probabilities, confidence

    def predict_batch(self, texts: list, batch_size: int = 64):
        self.model.eval()
        all_predicted_idxs = []
        all_probabilities = []
        all_confidences = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
            inputs.pop("token_type_ids", None)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            predicted_idxs = np.argmax(probabilities, axis=-1)
            confidences = probabilities[np.arange(len(probabilities)), predicted_idxs]
            
            all_predicted_idxs.extend(predicted_idxs.tolist())
            all_probabilities.extend(probabilities.tolist())
            all_confidences.extend(confidences.tolist())
            
        return all_predicted_idxs, all_probabilities, all_confidences

    def train_on_batch(self, texts, labels):
        """
        Online fine-tuning step.
        """
        self.model.train()
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        inputs.pop("token_type_ids", None)
        labels_tensor = torch.tensor(labels).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(**inputs, labels=labels_tensor)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        
        self.model.eval()
        return loss.item()

    def pretrain(self, train_texts, train_labels, batch_size=16, epochs=3, early_stopping_patience=None):
        # Build simple dataset holding raw texts and labels
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts, labels):
                self.texts = texts
                self.labels = labels
            def __len__(self):
                return len(self.texts)
            def __getitem__(self, idx):
                return self.texts[idx], self.labels[idx]
        
        # Split dataset if early stopping is requested
        val_texts, val_labels = [], []
        train_texts_split, train_labels_split = train_texts, train_labels
        
        if early_stopping_patience is not None and early_stopping_patience > 0:
            n_samples = len(train_texts)
            val_size = int(0.1 * n_samples)
            if val_size > 0:
                rng = np.random.default_rng(42)
                indices = np.arange(n_samples)
                rng.shuffle(indices)
                val_indices = indices[:val_size]
                train_indices = indices[val_size:]
                
                val_texts = [train_texts[i] for i in val_indices]
                val_labels = [train_labels[i] for i in val_indices]
                train_texts_split = [train_texts[i] for i in train_indices]
                train_labels_split = [train_labels[i] for i in train_indices]
                print(f"Split pretraining data: {len(train_texts_split)} train, {len(val_texts)} validation.")
        
        dataset = TextDataset(train_texts_split, train_labels_split)
        
        # Collate fn to tokenize per batch
        def collate_fn(batch):
            texts, labels = zip(*batch)
            inputs = self.tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs.pop("token_type_ids", None)
            labels_tensor = torch.tensor(labels)
            return inputs['input_ids'], inputs['attention_mask'], labels_tensor

        pin_mem = (self.device == "cuda")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_mem)
        
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in dataloader:
                b_input_ids, b_attention_mask, b_labels = [b.to(self.device) for b in batch]
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            
            if val_texts:
                self.model.eval()
                val_dataset = TextDataset(val_texts, val_labels)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_mem)
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_dataloader:
                        b_input_ids, b_attention_mask, b_labels = [b.to(self.device) for b in batch]
                        outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
                        val_loss += outputs.loss.item()
                avg_val_loss = val_loss / len(val_dataloader)
                print(f"Pretraining Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1} (patience={early_stopping_patience})")
                        break
            else:
                print(f"Pretraining Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")
                
        if val_texts and best_weights is not None:
            print("Restoring best weights based on validation loss.")
            best_weights_device = {k: v.to(self.device) for k, v in best_weights.items()}
            self.model.load_state_dict(best_weights_device)
            
        self.model.eval()
