import ollama
import json
import re
import os
import datetime
import numpy as np

class Tier2LLM:
    def __init__(self, model_name: str, url: str):
        self.model_name = model_name
        self.client = ollama.Client(host=url)

    def generate_prompt(self, text: str, categories: list, t1_label: str = None):
        t1_note = ""
        if t1_label is not None:
            t1_note = f'Note: The fast encoder (Tier 1) suggested "{t1_label}" for this text. Only override this if you are highly confident it is wrong.\n'

        prompt = f"""
Task: Classify the following news text into the most accurate category. 
Category Definitions:
- World: International news, state affairs, diplomacy, wars, and non-economic national events. (e.g., election results, peace talks, military strikes).
- Sports: Athletics, games, team news, athlete achievements, and competition results. (e.g., Olympic medals, football transfers, game scores).
- Business: Markets, finance, corporate actions, economic indicators, and trade. (Note: National economic reports like inflation or interest rates are 'Business' if they focus on market impact).
- Sci/Tech: Scientific research, consumer electronics, internet trends, space, and medicine. (e.g., new gadget releases, NASA missions, medical breakthrough).

Guidelines:
1. Provide a step-by-step reasoning (Chain of Thought) before the final label.
2. Only suggest a different category than you think Tier 1 might suggest if you are 100% certain. 
3. If the text is ambiguous, stick to the most obvious category.
Available Categories: {', '.join(categories)}

Examples:
Input Text: "United Nations officials meet to discuss the ongoing humanitarian crisis in Sudan."
Output:
labels: World
confidence: 0.98
reasoning: The focus is on a humanitarian crisis in Sudan and the involvement of the UN, which is an international diplomatic body. This fits 'World'.

Input Text: "The Lakers secured a narrow victory over the Celtics in a high-stakes NBA playoffs match."
Output:
labels: Sports
confidence: 0.99
reasoning: Mentions specific professional basketball teams and an NBA playoff game. This is clearly 'Sports'.

Input Text: "Global oil prices surged after major producers announced unexpected production cuts."
Output:
labels: Business
confidence: 0.96
reasoning: This text focuses on commodity pricing, global markets, and industrial production, which are core 'Business' topics.

Input Text: "Researchers have developed a new CRISPR-based method to treat genetic disorders more effectively."
Output:
labels: Sci/Tech
confidence: 0.97
reasoning: This discusses medical research and technological innovation in genetics, fitting the 'Sci/Tech' category.

---
{t1_note}Input Text: "{text}"
Output (structured key-value format):
"""
        return prompt

    def _clean_json_string(self, json_str: str) -> str:
        """
        Cleans common LLM JSON formatting issues, specifically invalid escape sequences.
        """
        # Replace raw backslashes that don't start a valid escape sequence
        # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        # We look for a backslash NOT followed by one of these characters
        cleaned = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', json_str)
        return cleaned

    def _parse_resilient(self, text: str) -> dict:
        """
        A highly robust structured key-value format parser designed specifically for LLM outputs.
        Handles multi-line values, different delimiters, and flexible key matching.
        """
        lines = text.strip().split('\n')
        result = {}
        current_key = None
        
        # Valid keys we expect from the LLM
        valid_keys = ['labels', 'confidence', 'reasoning', 'new_category_suggestion']
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Check if line starts with a known key followed by a colon
            # Regex handles case-insensitivity and optional whitespace
            match = re.match(r'^(' + '|'.join(valid_keys) + r')\s*:\s*(.*)', line_clean, re.IGNORECASE)
            
            if match:
                current_key = match.group(1).lower()
                value = match.group(2).strip()
                result[current_key] = value
            elif current_key:
                # If no key found, but we are in a multi-line section (like reasoning)
                # Append this line to the current value
                result[current_key] += "\n" + line_clean
        
        # --- Post-Processing & Normalization ---
        
        # 1. Normalize Labels (handle comma, semicolon, pipe, and single strings)
        if "labels" in result:
            labels_raw = str(result["labels"])
            # Split by common delimiters used by LLMs
            labels_list = re.split(r'[;,|]', labels_raw)
            # Clean and filter
            result["labels"] = [l.strip() for l in labels_list if l.strip()]
        else:
            result["labels"] = []

        # 2. Normalize Confidence (ensure it's a float)
        if "confidence" in result:
            try:
                # Extract the first float-like string found
                conf_val = str(result["confidence"])
                conf_match = re.search(r'0?\.\d+', conf_val)
                if conf_match:
                    result["confidence"] = float(conf_match.group())
                else:
                    result["confidence"] = 0.5
            except:
                result["confidence"] = 0.5
        else:
            result["confidence"] = 0.5
            
        # 3. Clean Reasoning (remove potential trailing dashes or markers)
        if "reasoning" in result:
            result["reasoning"] = result["reasoning"].strip()

        return result

    def predict(self, text: str, categories: list, t1_label: str = None, num_votes: int = 3):
        prompt = self.generate_prompt(text, categories, t1_label)
        votes = []
        raw_responses = []
        
        for _ in range(num_votes):
            try:
                # Use temperature 0.7 for voting diversity
                response = self.client.generate(model=self.model_name, prompt=prompt, options={"temperature": 0.7})
                res_text = response['response']
                raw_responses.append(res_text)
                result = self._parse_resilient(res_text)
                if result.get("labels"):
                    votes.append(result)
            except Exception as e:
                print(f"Voting error: {e}")
        
        if not votes:
            return None, "All votes failed"
            
        # Majority Vote Logic
        label_counts = {}
        for v in votes:
            for lbl in v["labels"]:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
        if not label_counts:
            return None, "No labels found in votes"
            
        # Pick top label
        best_label = max(label_counts, key=label_counts.get)
        
        # Calculate average confidence of votes for that label
        avg_conf = np.mean([v.get("confidence", 0.5) for v in votes if best_label in v["labels"]])
        
        final_result = {
            "labels": [best_label],
            "confidence": float(avg_conf),
            "reasoning": votes[0].get("reasoning", "Majority vote winner.")
        }
        
        return final_result, "\n---\n".join(raw_responses)

    def _log_error_response(self, response_text: str, error_msg: str):
        log_dir = "logs/tier2_errors"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"error_{timestamp}.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Error: {error_msg}\n")
            f.write("--- RAW RESPONSE ---\n")
            f.write(response_text)
            f.write("\n--- END RESPONSE ---\n")
