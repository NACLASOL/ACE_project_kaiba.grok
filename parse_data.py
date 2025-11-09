
import pdfplumber
import re
import json
import statistics
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
RUBRIC_FILE = 'internship.enhance_b2_article_writing_rubric.pdf'
EXAMPLES_FILE = 'internship.b2_writing_examples_scores.pdf'
TASK_PROMPT = """You have just seen the following advertisement in the university magazine:
Share with us what you think about the importance of physically attending classes in-person for university students instead of online classes. We're looking for articles about the benefits of studying in a classroom with a teacher. The best article will be published in our university magazine and the winner will receive a €200 gift card.
You have decided to contribute, Write an ARTICLE in which you:
• explain why physically attending classes can improve learning.
• mention why attending class at a university also provides the student with other
resources and facilities.
• suggest ways in which teachers can encourage students to go to class.
Give your article a title. Write your ARTICLE in 180-220 words."""

DEBUG_LOG = 'logs/parse_debug.log'
Path('logs').mkdir(exist_ok=True) # Create dir.
open(DEBUG_LOG, 'w').close() # Cleares the log.

def log_debug(message: str):
    with open(DEBUG_LOG, 'a', encoding='utf-8') as f:
        f.write(message + '\n')



# === RUBRIC EXTRACT (Returns formatted bullet list) ===
def extract_rubric(pdf_path: str) -> str:
    """
    Extracts the full rubric from the rubric PDF and returns a 
    clean string of bullets.
    """
    rubric_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = '\n'.join(page.extract_text() for page in pdf.pages)

        
        log_debug(f"Full rubric text:\n\n{full_text[:2000]}...\n") # Logs first 2000 chars for debugging.
        # Extract all level descriptors (Scores 5 to 1).
        # Pattern: "Score X (Level): Description until next Score or end".
        pattern = r'(\d) \(([^)]+)\)\s*(.*?)(?=\d \(|$)'
        matches = re.findall(pattern, full_text, re.DOTALL)
        
        bullets = []
        for level_num, level_label, desc in matches:
            clean_desc = re.sub(r'\s+', ' ', desc.strip()) # Normalize the whitespace.
            bullets.append(f"• level {level_num} ({level_label}): {clean_desc}")

        rubric_text = "\n".join(bullets)

        print(f"Extracted rubric: {len(bullets)} scoring levels (across all criteria).")
        log_debug(f"Extracted {len(bullets)} levels.\n")

    except Exception as e:
        print(F"Error parsing rubric: {e}")
        log_debug(f"Error: {e}")
        # This is the fallback for the 5 criteria.
        fallback = "• Level 5 (Excellent): Excellent performance. \n• Level 4 (Good): Good performance. \n• Score 3 (Satisfactory): Satisfactory performance. \n• Score 2 (Limited): Limited performance. \n• Score 1 (Inadequate): Inadequate performance"
        rubric_text = fallback * 5 # Repeat for TA, CC, GR, LR, OWP.

    return rubric_text


# === STUDENT EXAMPLE EXTRACT + HUMAN SCORES ===
def extract_examples(pdf_path: str, rubric_bullets: str) -> list:
    """
    Parses the student examples PDF and returns a list of samples.
    Each sample includes:
    - input: Full prompt (task + student article + rubric).
    - ground_truth: Dictionary of human scores.
    """
    samples = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]

        current_article = ""
        current_title = ""
        title_pending = False # Flags to capture title on next line.
        in_article = False
        example_num = 0

        for page_idx, text in enumerate(pages_text):
            lines = text.split('\n')
            for line_num, line in enumerate(lines):
                line = line.strip()

                # Detect a new example.
                if re.match(r'^Example \d+', line, re.IGNORECASE):
                    if current_article and example_num > 0:
                        # Save previous.
                        samples.append(_finalize_sample(current_title, current_article, rubric_bullets))
                    example_num += 1
                    current_title = ""
                    current_article = ""
                    in_article = True
                    title_pending = True # Expect a title on next non-empty line.
                    log_debug(f"Detected Example {example_num} on page {page_idx+1}, line {line_num}")
                    continue

                if in_article and title_pending and line:
                    current_title = line
                    title_pending = False
                    log_debug(f"Captured title: '{current_title}' for Example {example_num}")
                    continue

                # Detect score table (end of an article).
                if re.search(r'Task appropriateness\s*\d', line):
                    in_article = False
                    title_pending = False
                    # Extract scores from this + next 10 lines/pages lines. Increased range is for safety.
                    score_lines = []
                    for i in range(page_idx, min(page_idx + 10, len(pages_text))):
                        score_lines.extend(pages_text[i].split('\n'))

                    scores = _extract_scores_from_lines(score_lines)
                    if len(scores) >= 4:
                        if 'OWP' not in scores:
                            avg = statistics.mean(scores.values())
                            scores['OWP'] = round(avg) # Rounds to nearest whole number for CEFR alignment.
                    
                        samples.append(_finalize_sample(current_title, current_article, rubric_bullets, scores))
                        log_debug(f"Extracted scores for Example {example_num}: {scores}\n")
                    else:
                        log_debug(f"Incomplete scores for Example {example_num}: {scores}")
                    
                    current_article = ""
                    continue
                
                # Append to article if in_article and after title.
                if in_article and not title_pending and line:
                    current_article += line + " "

        # Save the last article if it's pending.        
        if current_article and example_num > 0:
            samples.append(_finalize_sample(current_title, current_article, rubric_bullets))

    except Exception as e:
        print(f"Error parsing examples: {e}")
        log_debug(f"Error: {e}")
    
    print(f"Succesfully extracted {len(samples)} complete student examples.")
    return samples

def _extract_scores_from_lines(lines: list) -> dict:
    """Helper that extracts TA, CC, GR, LR scores from table lines"""
    scores = {}
    patterns = {
        'TA': r'Task appropriateness\s*(\d(?:\.\d)?)',
        'CC': r'Coherence and cohesion\s*(\d(?:\.\d)?)',
        'GR': r'Grammatical range and accuracy\s*(\d(?:\.\d)?)',
        'LR': r'Lexical range and accuracy\s*(\d(?:\.\d)?)',
        'OWP': r'Overall written production\s*(\d(?:\.\d)?)',
    }

    text = " ".join(lines)
    log_debug(f"Score text block: {text[:500]}...") # Debug partial.

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE) # Case-insensitive for safety.
        if match:
            scores[key] = float(match.group(1))

    return scores

def _finalize_sample(title: str, article: str, rubric_bullets: str, scores: Dict[str, float] = None) -> Dict:
    """Builds full input prompt and pairs with ground truth"""
    article = article.strip()
    full_input = f"""TASK: {TASK_PROMPT}

STUDENT ARTICLE TITLE: {title}
STUDENT ARTICLE:
{article}

RUBRIC:
{rubric_bullets}

INSTRUCTIONS: You are grading a B2 English article. Analyze step-by-step in 'reasoning', reference useful playbook bullets in 'bullet_ids', and put socres in 'final_answer' as RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0.
OUTPUT RAW JSON ONLY: {{"reasoning": "your analysis with \\n for breaks", "bullet_ids": ["id1", "id2"], "final_answer": "RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0"}}. NO Markdown (no '''json), NO extra text, No explanations outside JSON. Use \\n for line breaks in reasoning. Scores: whole floats [1.0-5.0]."""


    if scores and len(scores) >= 4 and 'OWP' not in scores:
        avg = statistics.mean([scores.get(k, 3.0) for k in ['TA', 'CC', 'GR', 'LR']])
        scores['OWP'] = round(avg) # Whole number. 


    ground_truth = scores or {
        'TA' : 0.0, 'CC' : 0.0, 'GR' : 0.0, 'LR' : 0.0, 'OWP' : 0.0
    }
# REMOVED FOR DEBUG PURPOSES. UN-COMMENT AND RETURN TO ORIGINAL LINE: ... = scores ...

    return {
        'input' : full_input.strip(),
        'ground_truth' : ground_truth
    }

print("Parsing rubric...")
rubric_bullets = extract_rubric(RUBRIC_FILE)

print("Parsing student examples...")
samples = extract_examples(EXAMPLES_FILE, rubric_bullets)

# === MAIN EXECUTION ===
if __name__ == "__main__":

    # Save the file.
    output_file = 'upv_samples.json'
    json.dump(samples, open(output_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(samples)} samples to {output_file}.")
    print(f"Debug log saved to {DEBUG_LOG} for inspection")


train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)
one_sample = [samples[0]]
json.dump(train_samples, open('logs/train_samples.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
json.dump(test_samples, open('logs/test_samples.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
json.dump(one_sample, open('logs/one_sample.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)