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
Task: Classify the following text into one or more of the specified categories. provide CoT reasoning.
Available Categories: {', '.join(categories)}

Input Text: "{text}"

Output Format (STRICT JSON):
{{
  "labels": ["Category1", "Category2"],
  "confidence": 0.95,
  "reasoning": "Explain why this label was chosen based on the text.",
  "new_category_suggestion": "Optional category name if none of the above fit perfectly"
}}
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

    def predict(self, text: str, categories: list):
        prompt = self.generate_prompt(text, categories)
        response_text = ""
        try:
            response = self.client.generate(model=self.model_name, prompt=prompt)
            response_text = response['response']
            
            # Extract JSON from response (handling potential markdown formatting)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    result = json.loads(json_str)
                    return result, response_text
                except json.JSONDecodeError:
                    # Try cleaning the string and parsing again
                    cleaned_json = self._clean_json_string(json_str)
                    try:
                        result = json.loads(cleaned_json)
                        return result, response_text
                    except json.JSONDecodeError as e:
                        self._log_error_response(response_text, str(e))
                        return None, response_text
            else:
                return None, response_text
        except Exception as e:
            print(f"Error in Tier 2 LLM: {e}")
            if response_text:
                self._log_error_response(response_text, str(e))
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
