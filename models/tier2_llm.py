import ollama
import json
import re
import os
import datetime

class Tier2LLM:
    def __init__(self, model_name: str, url: str):
        self.model_name = model_name
        self.client = ollama.Client(host=url)

    def generate_prompt(self, text: str, categories: list):
        prompt = f"""
Task: Classify the following news text into the most accurate category. 
Category Definitions:
- World: Global affairs, international relations, diplomacy, foreign conflicts, and events occurring outside the US or of global significance.
- Sports: Coverage of professional/amateur athletics, matches, team news, athlete profiles, and major tournaments (NBA, NFL, FIFA, etc.).
- Business: Finance, stock markets, corporate mergers, economic policy, trade, and industry trends.
- Sci/Tech: Scientific discoveries, technological innovations, software/hardware releases, space exploration, and medical research.

Guidelines:
1. Provide a step-by-step reasoning (Chain of Thought) before the final label.
2. If the text fits multiple categories, choose the primary focus.
3. Be decisive but only report high confidence if the evidence is clear.
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
Input Text: "{text}"
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

    def predict(self, text: str, categories: list):
        prompt = self.generate_prompt(text, categories)
        response_text = ""
        try:
            response = self.client.generate(model=self.model_name, prompt=prompt)
            response_text = response['response']
            
            # 1. Attempt Resilient Parsing (Primary)
            result = self._parse_resilient(response_text)
            
            # Check if we got at least labels and confidence
            if result.get("labels") and "confidence" in result:
                return result, response_text
                
            # 2. Fallback: Check if it's JSON anyway (Legacy Support)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(self._clean_json_string(json_match.group()))
                    if isinstance(result.get("labels"), str):
                        result["labels"] = [result["labels"]]
                    return result, response_text
                except:
                    pass
            
            # 3. If everything fails, log and return None
            self._log_error_response(response_text, "Failed to parse as structured key-value format or JSON")
            return None, response_text
            
        except Exception as e:
            print(f"Error in Tier 2 LLM: {e}")
            self._log_error_response(response_text if response_text else "NO_RESPONSE", str(e))
            return None, str(e)

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
