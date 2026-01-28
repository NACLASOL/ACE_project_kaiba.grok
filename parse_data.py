
import pdfplumber
import re
import json
import statistics
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split

from constants import RUBRIC_FILE, EXAMPLES_FILE, TASK_PROMPT, PARSE_DEBUG_LOG, LOG_DIR, LATEST_PLAYBOOK_FILE

DEBUG_LOG = PARSE_DEBUG_LOG
Path(LOG_DIR / "samples").mkdir(parents=True, exist_ok=True)
open(DEBUG_LOG, 'w').close() # Cleares the log.

def log_debug(message: str):
    with open(DEBUG_LOG, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


# === RUBRIC EXTRACT ===
def extract_rubric(pdf_path: str) -> str:
    """
    Extracts the full rubric from the NEW structured rubric PDF format.
    
    NEW FORMAT (internship.enhance_b2_article_writing_rubric_structured.pdf):
    - Category Name (no numbering, no "Level ... Performance Descriptors" header)
    - 5 (Excellent)
    - • Bullet points describing level 5
    - 4 (Good)
    - • Bullet points describing level 4
    - etc.
    
    Returns a formatted string with category headers and level descriptions.
    """
    rubric_lines = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = '\n'.join(page.extract_text() or "" for page in pdf.pages)
            log_debug(f"FULL RUBRIC TEXT (first 3000 chars):\n{full_text[:3000]}\n{'='*80}\n")
        
        # Pre-clean: normalize whitespace but preserve line breaks for structure
        full_text = re.sub(r'\n\s*\n', '\n', full_text)  # Remove multiple blank lines
        
        # Define the 5 expected categories in order (matching new rubric exactly)
        expected_categories = [
            "Task Achievement",
            "Cohesion and Coherence", 
            "Grammatical Accuracy and Range",
            "Lexical Accuracy and Range",
            "Overall Written Production"
        ]
        
        # Build a regex pattern that captures each category section
        # Pattern: Category name, followed by all level blocks until next category or end
        category_sections = []
        
        for i, category in enumerate(expected_categories):
            # Escape special regex characters in category name
            escaped_category = re.escape(category)
            
            # Build lookahead for next category (or end of string)
            if i < len(expected_categories) - 1:
                next_category = re.escape(expected_categories[i + 1])
                lookahead = f"(?={next_category}|\\Z)"
            else:
                lookahead = "\\Z"  # Last category goes to end
            
            # Pattern: category name followed by content until next category
            pattern = f"{escaped_category}\\s*(.+?){lookahead}"
            match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                category_content = match.group(1).strip()
                category_sections.append((category, category_content))
                log_debug(f"Found category: {category} with {len(category_content)} chars")
            else:
                log_debug(f"WARNING: Category '{category}' not found in rubric")
        
        # Now extract levels from each category section
        for category_name, content in category_sections:
            rubric_lines.append(f"\n**{category_name}**\n")
            
            # Extract level blocks: "5 (Excellent)", "4 (Good)", etc.
            # Pattern matches: number, parenthesized label, followed by bullet points
            level_pattern = r'(\d)\s*\(([A-Za-z]+)\)\s*((?:•[^\n]+(?:\n(?!•?\d\s*\().*?)*)*)'
            levels = re.findall(level_pattern, content, re.DOTALL)
            
            if not levels:
                log_debug(f"WARNING: No levels found for category '{category_name}'")
                continue
            
            for lvl_num, lvl_name, desc in levels:
                # Clean up descriptor text
                desc = desc.strip()
                
                # Remove page numbers that might have been captured
                desc = re.sub(r'\n\s*\d+\s*$', '', desc)
                
                # Replace multiple spaces with single space
                desc = re.sub(r'\s+', ' ', desc)
                
                # Ensure bullet points are preserved and formatted consistently
                desc = re.sub(r'•\s*', '• ', desc)
                
                rubric_lines.append(f"• Level {lvl_num} ({lvl_name}): {desc}")
                
            log_debug(f"Extracted {len(levels)} levels for '{category_name}'")
        
        final_rubric = "\n".join(rubric_lines).strip()
        log_debug(f"\nFinal RUBRIC:\n{final_rubric}\n{'='*80}\n")
        
        print(f"✅ Extracted rubric with {len(category_sections)} categories")
        return final_rubric
        
    except Exception as e:
        print(f"❌ Error parsing rubric: {e}")
        log_debug(f"Error: {e}")
        
        # Fallback rubric with 5 categories (using correct category names)
        categories = [
            "Task Achievement", 
            "Cohesion and Coherence", 
            "Grammatical Accuracy and Range",
            "Lexical Accuracy and Range", 
            "Overall Written Production"
        ]
        
        fallback = ""
        for cat in categories:
            fallback += f"\n**{cat}**\n"
            fallback += "• Level 5 (Excellent): Excellent performance.\n"
            fallback += "• Level 4 (Good): Good performance.\n"
            fallback += "• Level 3 (Satisfactory): Satisfactory performance.\n"
            fallback += "• Level 2 (Limited): Limited performance.\n"
            fallback += "• Level 1 (Inadequate): Inadequate performance.\n"
        
        print("⚠️  Using fallback rubric due to extraction error")
        return fallback.strip()
    

# === STUDENT EXAMPLE EXTRACT + HUMAN SCORES ===
def extract_examples(pdf_path: str, rubric_bullets: str) -> List[Dict]:
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

                # Capture title if pending and line is non-empty.
                if in_article and title_pending and line:
                    current_title = line
                    title_pending = False
                    log_debug(f"Captured title: '{current_title}' for Example {example_num}")
                    continue

                # Detect score table (end of an article).
                if re.search(r'Task appropriateness\s*\d', line):
                    in_article = False
                    title_pending = False
                    # Extract scores from this + next 10 lines/pages. Increased range is for safety.
                    score_lines = []
                    for i in range(page_idx, min(page_idx + 10, len(pages_text))):
                        score_lines.extend(pages_text[i].split('\n'))

                    scores = _extract_scores_from_lines(score_lines)
                    if len(scores) >= 4:    # Valid if at least 4 scores.
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

def _extract_scores_from_lines(lines: list) -> Dict[str, float]:
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
"""if __name__ == "__main__":

    # Save the file.
    output_file = 'logs/samples/upv_samples.json'
    json.dump(samples, open(output_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(samples)} samples to {output_file}.")
    print(f"Debug log saved to {DEBUG_LOG} for inspection.")"""

one_sample = [samples[0]]
json.dump(samples, open('logs/samples/train_samples.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)