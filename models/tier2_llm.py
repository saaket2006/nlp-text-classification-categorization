import ollama
import json
import re
import os
import datetime
import numpy as np

class Tier2LLM:
    def __init__(self, model_name: str, url: str, num_votes: int = 3):
        self.model_name = model_name
        self.client = ollama.Client(host=url)
        self.num_votes = num_votes
        self.offline = True
        self.definitions = ""
        self.examples = ""
        self.cache = {}
        self.sample_counter = 0
        self.current_seed = None
        try:
            self.client.list()
            self.offline = False
        except Exception:
            self.offline = True

    def setup_dataset(self, dataset_name: str, categories: list):
        self.dataset_name = dataset_name
        self.categories = categories
        self.definitions = ""
        self.examples = ""
        self.sample_counter = 0
        self.cache = {}
        
        # Load local text-based cache if available
        os.makedirs(f"logs/{dataset_name}", exist_ok=True)
        local_cache_path = f"logs/{dataset_name}/llm_cache.json"
        if os.path.exists(local_cache_path):
            try:
                with open(local_cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"Loaded persistent LLM cache for {dataset_name} ({len(self.cache)} entries).")
            except Exception as e:
                print(f"Warning: Failed to load local LLM cache: {e}")


        
        # 1. Check if we have pre-defined config
        if dataset_name == "ag_news":
            self.definitions = """- World: International news, politics, war, disasters, terrorism, elections, public figures' deaths, trials, and international diplomatic events. (Note: General news stories about sports figures' deaths or religious observances like Yom Kippur in sports are classified as World).
- Sports: Professional and amateur athletics, games, league news, team scores, and match results. (Note: Video games and computer software/hardware games belong to Sci/Tech, not Sports).
- Business: Economy, trade, stocks, markets, finance, general corporate earnings, mergers, and business acquisitions. (Note: News about tech/telecom companies' corporate actions belongs to Sci/Tech).
- Sci/Tech: Science, research, technology, consumer electronics, software, hardware, space exploration (NASA), environmental treaties/policy (Kyoto Protocol), urbanization, and tech/telecom company news (including acquisitions, stock movement, or product prices of tech companies like Microsoft, Intel, Cisco, HP, Sprint, etc.)."""
            self.examples = """Input Text: "United Nations officials meet to discuss the ongoing humanitarian crisis in Sudan."
Output:
labels: World
confidence: 0.98
reasoning: The focus is on a humanitarian crisis in Sudan and the involvement of the UN. This fits 'World'.

Input Text: "The Lakers secured a narrow victory over the Celtics in a high-stakes NBA playoffs match."
Output:
labels: Sports
confidence: 0.99
reasoning: Mentions professional basketball teams and an NBA playoff game. This is clearly 'Sports'.

Input Text: "Global oil prices surged after major producers announced unexpected production cuts."
Output:
labels: Business
confidence: 0.96
reasoning: Focuses on commodity pricing, global markets, and industrial production, which are core 'Business' topics.

Input Text: "Russia expects its parliament to ratify the Kyoto Protocol this month, allowing the climate change treaty to come into force."
Output:
labels: Sci/Tech
confidence: 0.95
reasoning: This is about the Kyoto Protocol and climate change treaty, which are environmental science policy and classified as Sci/Tech.

Input Text: "Dodgers Nip Giants 3-2 in Crucial Series. Shawn Green will miss Saturday to observe the Jewish holiday Yom Kippur."
Output:
labels: World
confidence: 0.95
reasoning: While it mentions a baseball game, the text includes a significant religious angle (observing Yom Kippur), which is classified under World news in this corpus.

Input Text: "Level 3 today announced that it has purchased Sprint's wholesale dial-up Internet access business for $34 million in cash."
Output:
labels: Business
confidence: 0.96
reasoning: This is a corporate acquisition and cash purchase of a business unit, which falls under 'Business' in the corpus."""
            return

        elif dataset_name == "dbpedia_14":
            self.definitions = """- Company: Companies, startups, corporations, businesses, and commercial entities (e.g., Microsoft, Ford, retail shops).
- EducationalInstitution: Schools, universities, colleges, academies, and educational organizations (e.g., Harvard University, Oxford).
- Artist: Musicians, painters, writers, actors, directors, sculptors, and other creative professionals (e.g., Shakespeare, Mozart).
- Athlete: Sports players, racers, Olympic competitors, and athletic figures (e.g., Michael Jordan, Serena Williams).
- OfficeHolder: Politicians, government officials, kings, queens, presidents, prime ministers, and judges.
- MeanOfTransportation: Vehicles, planes, trains, spacecraft, ships, and other transportation methods.
- Building: Architectural structures, airports, museums, stadiums, bridges, and skyscrapers (e.g., Eiffel Tower, Empire State Building).
- NaturalPlace: Oceans, rivers, mountains, lakes, forests, caves, and geographical features.
- Village: Small towns, settlements, villages, and rural municipalities.
- Animal: Non-human mammals, birds, insects, reptiles, fish, and other fauna.
- Plant: Flowers, trees, plants, seeds, and other flora.
- Album: Music albums, EPs, soundtracks, and audio recordings.
- Film: Movies, motion pictures, documentaries, and cinema releases.
- WrittenWork: Books, novels, magazines, newspapers, plays, and written publications (e.g., Harry Potter)."""
            self.examples = """Input Text: "Microsoft Corporation is an American multinational technology corporation."
Output:
labels: Company
confidence: 0.99
reasoning: Discusses Microsoft, which is a major commercial company.

Input Text: "William Shakespeare was an English playwright, poet, and actor."
Output:
labels: Artist
confidence: 0.98
reasoning: Shakespeare was a playwright and poet, which is classified as an Artist.

Input Text: "The Boeing 747 is a large, long-range wide-body airliner designed and manufactured by Boeing Commercial Airplanes."
Output:
labels: MeanOfTransportation
confidence: 0.99
reasoning: Discusses the Boeing 747, which is a commercial airplane and thus a mean of transportation.

Input Text: "Harry Potter and the Philosopher's Stone is a fantasy novel written by British author J. K. Rowling."
Output:
labels: WrittenWork
confidence: 0.98
reasoning: It is a fantasy novel, which is a written work."""
            return

        elif dataset_name in ["imdb", "amazon_polarity", "yelp_polarity", "sst2", "glue/sst2"]:
            self.definitions = """- negative: Expresses dissatisfaction, bad experiences, criticism, disappointment, or overall negative sentiment.
- positive: Expresses satisfaction, good experiences, praise, enjoyment, approval, or overall positive sentiment."""
            self.examples = """Input Text: "The acting was terrible, the plot made no sense, and I walked out halfway through."
Output:
labels: negative
confidence: 0.99
reasoning: The user expresses strong dissatisfaction and describes the movie negatively ("terrible", "no sense").

Input Text: "Absolutely brilliant! The cinematography was stunning and the performance was Oscar-worthy."
Output:
labels: positive
confidence: 0.99
reasoning: The user praises the movie with words like "absolutely brilliant" and "stunning", which is highly positive."""
            return

        elif dataset_name == "emotion":
            self.definitions = """- sadness: Expresses grief, sorrow, disappointment, depression, or feeling down.
- joy: Expresses happiness, excitement, satisfaction, delight, or pleasure.
- love: Expresses affection, warmth, liking, attraction, or strong positive attachment.
- anger: Expresses annoyance, frustration, fury, resentment, or hostility.
- fear: Expresses anxiety, worry, apprehension, panic, or terror.
- surprise: Expresses astonishment, shock, amazement, or wonder at something unexpected."""
            self.examples = """Input Text: "I feel so lonely and empty inside today."
Output:
labels: sadness
confidence: 0.98
reasoning: Mentions feeling "lonely" and "empty", which are clear indicators of sadness.

Input Text: "I just got accepted into my dream college! I can't stop smiling!"
Output:
labels: joy
confidence: 0.99
reasoning: Getting accepted into a dream college and not being able to stop smiling are strong indicators of joy.

Input Text: "I am so incredibly angry that they lied to me after all this time."
Output:
labels: anger
confidence: 0.98
reasoning: The user explicitly states they are "incredibly angry" about being lied to.

Input Text: "My heart is pounding and I'm terrified to walk down the dark alley alone."
Output:
labels: fear
confidence: 0.99
reasoning: Mentions a pounding heart and being "terrified", which are classic physiological and emotional signs of fear."""
            return

        # 2. Dynamic generation using Ollama if online
        if not self.offline:
            try:
                print(f"Generating dynamic prompt definitions and examples for dataset '{dataset_name}' using {self.model_name}...")
                gen_prompt = f"""You are an expert AI data annotator.
We need to classify texts into the following categories: {categories}.
Please write a concise 1-sentence definition/description for each category to help a classifier.
Then, write 1 typical short example sentence (text) and its correct classification for any 2 of the categories.

Format your response exactly as follows:
DEFINITIONS:
- <Category1>: <definition1>
- <Category2>: <definition2>
...

EXAMPLES:
Input Text: "<Example text 1>"
Output:
labels: <CategoryA>
confidence: 0.95
reasoning: <Brief explanation>

Input Text: "<Example text 2>"
Output:
labels: <CategoryB>
confidence: 0.95
reasoning: <Brief explanation>
"""
                response = self.client.generate(model=self.model_name, prompt=gen_prompt, options={"temperature": 0.1})
                response_text = response['response']
                
                # Parse the response text to split DEFINITIONS and EXAMPLES
                if "DEFINITIONS:" in response_text and "EXAMPLES:" in response_text:
                    parts = response_text.split("EXAMPLES:")
                    self.definitions = parts[0].replace("DEFINITIONS:", "").strip()
                    self.examples = parts[1].strip()
                    print(f"Successfully generated dynamic definitions and examples!")
                    return
            except Exception as e:
                print(f"Failed to dynamically generate definitions via Ollama: {e}")
                
        # 3. Fallback: Generic generation
        print("Using generic fallback for definitions and examples.")
        defs = []
        for cat in categories:
            defs.append(f"- {cat}: Texts that belong to or are related to the category '{cat}'.")
        self.definitions = "\n".join(defs)
        
        exs = []
        for cat in categories[:2]:
            exs.append(f"""Input Text: "This is a representative example text specifically about {cat} and its related topics."
Output:
labels: {cat}
confidence: 0.95
reasoning: The text explicitly mentions and focuses on topics related to {cat}.""")
        self.examples = "\n\n".join(exs)

    def generate_prompt(self, text: str, categories: list, t1_label: str = None):
        t1_note = ""
        if t1_label is not None:
            if self.dataset_name == "dbpedia_14":
                t1_note = (
                    f'Note: The Tier 1 encoder predicted "{t1_label}" for this text.\n'
                    f'Decide whether to keep this prediction or override it. You should keep "{t1_label}" if it is reasonable.\n'
                    f'However, you MUST override "{t1_label}" if it is a clear mistake. For example:\n'
                    f'- If the text describes a plant/seed/flower, and Tier 1 predicted "{t1_label}" (such as "Animal"), override it to "Plant".\n'
                    f'- If the text describes a writer/author/musician/director, and Tier 1 predicted "{t1_label}" (such as "OfficeHolder"), override it to "Artist".\n'
                    f'- If the text describes a village/town/settlement, and Tier 1 predicted "{t1_label}" (such as "Building"), override it to "Village".\n\n'
                    f'Please follow these DBpedia conventions to avoid false overrides:\n'
                    f'- Do NOT override "{t1_label}" if it is "OfficeHolder" and the text is about a professor, researcher, or academic.\n'
                    f'- Do NOT override "{t1_label}" if it is "Company" and the text is about a collective organization, association, non-profit, or society.\n'
                    f'- Do NOT override "{t1_label}" if it is "Building" and the text is about an athenaeum, museum, library, or cultural institution.\n'
                    f'- Do NOT override "{t1_label}" if it is "WrittenWork" and the text is about a scholarly society or journal.\n'
                    f'- If the text is extremely short (e.g. just a name like "John R.") or has zero context, never override.\n'
                )
            else:
                t1_note = f'Note: The fast encoder (Tier 1) suggested "{t1_label}" for this text. Only override this if you are highly confident it is wrong.\n'

        guidelines = """1. Provide a step-by-step reasoning (Chain of Thought) before the final label.
2. Only suggest a different category than you think Tier 1 might suggest if you are 100% certain. 
3. If the text is ambiguous, stick to the most obvious category."""

        if self.dataset_name == "dbpedia_14":
            guidelines = """1. Provide a step-by-step reasoning (Chain of Thought) before the final label.
2. Only override the Tier 1 prediction if you are 100% certain it is completely wrong and violates the definitions.
3. If the text is ambiguous or fits DBpedia conventions mentioned below, stick to the Tier 1 prediction."""

        prompt = f"""
Task: Classify the following text into the most accurate category. 
Category Definitions:
{self.definitions}

Guidelines:
{guidelines}
Available Categories: {', '.join(categories)}

Examples:
{self.examples}

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
        Handles full JSON parsing, multi-line values, different delimiters, and flexible key matching.
        """
        # Try full JSON parsing first
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                json_str = self._clean_json_string(json_str)
                parsed_json = json.loads(json_str)
                
                normalized = {}
                for k, v in parsed_json.items():
                    key_clean = k.strip().lower()
                    normalized[key_clean] = v
                
                if "labels" in normalized or "label" in normalized:
                    result = {}
                    lbl_val = normalized.get("labels", normalized.get("label"))
                    if isinstance(lbl_val, list):
                        result["labels"] = [str(x).strip() for x in lbl_val if str(x).strip()]
                    elif lbl_val:
                        labels_list = re.split(r'[;,|]', str(lbl_val))
                        result["labels"] = [l.strip() for l in labels_list if l.strip()]
                    else:
                        result["labels"] = []
                    
                    conf_val = normalized.get("confidence", 0.5)
                    try:
                        result["confidence"] = float(conf_val)
                    except:
                        result["confidence"] = 0.5
                        
                    result["reasoning"] = str(normalized.get("reasoning", "")).strip()
                    return result
        except Exception:
            pass

        # Fallback to line-by-line key-value parsing
        lines = text.strip().split('\n')
        result = {}
        current_key = None
        valid_keys = ['labels', 'label', 'confidence', 'reasoning', 'new_category_suggestion']
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Check if line matches optional prefix/quotes + key + optional quotes + colon + value
            match = re.match(r'^[^\w"\']*(' + '|'.join(valid_keys) + r')[^\w"\']*\s*:\s*(.*)', line_clean, re.IGNORECASE)
            
            if match:
                current_key = match.group(1).lower()
                value = match.group(2).strip()
                # Clean value from surrounding quotes/commas/brackets
                value = re.sub(r'^["\'\[\s]+|["\'\]\s,]+$', '', value)
                result[current_key] = value
            elif current_key:
                line_val = line_clean
                if line_val.endswith('"') or line_val.endswith('",') or line_val.endswith("'") or line_val.endswith("',"):
                    line_val = re.sub(r'^["\'\s]+|["\'\s,]+$', '', line_val)
                result[current_key] += "\n" + line_val

        # Normalize label to labels
        if "label" in result and "labels" not in result:
            result["labels"] = result["label"]
            
        # Post-Process labels
        if "labels" in result:
            labels_raw = str(result["labels"])
            labels_list = re.split(r'[;,|]', labels_raw)
            result["labels"] = [l.strip() for l in labels_list if l.strip()]
        else:
            result["labels"] = []

        # Post-Process confidence
        if "confidence" in result:
            try:
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
            
        if "reasoning" in result:
            result["reasoning"] = result["reasoning"].strip()
            
        return result


    def normalize_label(self, label: str, categories: list) -> str:
        label_clean = label.strip().lower()
        # 1. Exact match (case insensitive)
        for cat in categories:
            if cat.lower() == label_clean:
                return cat
                
        # 2. Sentiment specific mapping (positive/pos, negative/neg)
        sentiment_pos = ["positive", "pos", "positive sentiment", "p"]
        sentiment_neg = ["negative", "neg", "negative sentiment", "n"]
        
        if label_clean in sentiment_pos:
            for cat in categories:
                if cat.lower() in sentiment_pos:
                    return cat
        if label_clean in sentiment_neg:
            for cat in categories:
                if cat.lower() in sentiment_neg:
                    return cat
                    
        # 3. Fallback: prefix/substring match
        for cat in categories:
            if cat.lower().startswith(label_clean) or label_clean.startswith(cat.lower()):
                return cat
                
        return label  # keep original if no match


    def save_cache(self):
        if not hasattr(self, "dataset_name") or not self.dataset_name:
            return
        local_cache_path = f"logs/{self.dataset_name}/llm_cache.json"
        try:
            with open(local_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save LLM cache: {e}")

    def predict(self, text: str, categories: list, t1_label: str = None, sample_idx: int = None, num_votes: int = None):
        if sample_idx is not None:
            idx = sample_idx
        else:
            self.sample_counter += 1
            idx = self.sample_counter - 1
        
        # Check in-memory cache keyed by text
        if text in self.cache:
            return self.cache[text]
            
        # 3. Call actual prediction
        res = self._real_predict(text, categories, t1_label, num_votes)
        self.cache[text] = res
        self.save_cache()
        return res

    def _real_predict(self, text: str, categories: list, t1_label: str = None, num_votes: int = None):
        if num_votes is None:
            num_votes = self.num_votes
        prompt = self.generate_prompt(text, categories, t1_label)
        votes = []
        raw_responses = []
        
        for _ in range(num_votes):
            try:
                # Use temperature 0.1 for more stable classification
                response = self.client.generate(model=self.model_name, prompt=prompt, options={"temperature": 0.1})
                res_text = response['response']
                raw_responses.append(res_text)
                result = self._parse_resilient(res_text)
                if result.get("labels"):
                    votes.append(result)
            except Exception as e:
                print(f"Voting error: {e}")
        
        if not votes:
            return None, "All votes failed"
            
        # Majority Vote Logic with Label Normalization
        label_counts = {}
        for v in votes:
            for lbl in v["labels"]:
                norm_lbl = self.normalize_label(lbl, categories)
                label_counts[norm_lbl] = label_counts.get(norm_lbl, 0) + 1
        
        if not label_counts:
            return None, "No labels found in votes"
            
        # Pick top label
        best_label = max(label_counts, key=label_counts.get)
        
        # Calculate average confidence of votes for that label (considering normalized matches)
        avg_conf = np.mean([
            v.get("confidence", 0.5) 
            for v in votes 
            if any(self.normalize_label(x, categories) == best_label for x in v["labels"])
        ])
        
        final_result = {
            "labels": [best_label],
            "confidence": float(avg_conf),
            "reasoning": votes[0].get("reasoning", "Majority vote winner."),
            "vote_agreement": label_counts[best_label] / len(votes) if votes else 0.0
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
