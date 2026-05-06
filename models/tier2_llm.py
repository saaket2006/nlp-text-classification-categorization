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
Task: Classify the following text into one or more of the specified categories. Provide CoT (Chain of Thought) reasoning.
Available Categories: {', '.join(categories)}

Examples:
Input Text: "United Nations officials meet to discuss the ongoing humanitarian crisis in Sudan."
Output: {{
  "labels": ["World"],
  "confidence": 0.98,
  "reasoning": "The text mentions international diplomatic bodies (UN) and humanitarian crises in a specific country, which fits the 'World' category."
}}

Input Text: "The Lakers secured a narrow victory over the Celtics in a high-stakes NBA playoffs match."
Output: {{
  "labels": ["Sports"],
  "confidence": 0.99,
  "reasoning": "Mentions professional basketball teams (Lakers, Celtics) and sporting events (NBA playoffs), clearly belonging to 'Sports'."
}}

Input Text: "Global oil prices surged after major producers announced unexpected production cuts."
Output: {{
  "labels": ["Business"],
  "confidence": 0.96,
  "reasoning": "Discusses global market prices and production announcements by industry producers, which is characteristic of 'Business' news."
}}

Input Text: "A new study reveals that advanced machine learning algorithms can predict solar flares with 90% accuracy."
Output: {{
  "labels": ["Sci/Tech"],
  "confidence": 0.97,
  "reasoning": "Focuses on scientific research, machine learning technology, and solar phenomena, aligning with 'Sci/Tech'."
}}

---
Input Text: "{text}"
Output (STRICT JSON):
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
