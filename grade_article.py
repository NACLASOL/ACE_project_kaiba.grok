import json
import os
import re
from dotenv import load_dotenv
from parse_data import extract_rubric
from datetime import datetime
from ace import *
from opik import track
from pathlib import Path
from upv_grader import *

load_dotenv()

RUBRIC_FILE = "parse/internship.enhance_b2_article_writing_rubric.pdf"

# Load evolved playbook (from last adaptation_summary.json)
LOG_DIR = Path("logs")
summary = json.load(open(LOG_DIR / "adaptation_summary.json"))
latest_playbook = Playbook.load_from_file("logs/latest_playbook.json")

PLAYBOOK_PROMPT = "\n".join(str(latest_playbook.bullets())) if latest_playbook else "No evolved playbook found - using default" # Convert to string.

RUBRIC = extract_rubric(RUBRIC_FILE)

@track(project_name=os.getenv("OPIK_PROJECT_NAME"))
def grade_article(article_title: str, article_text: str, ) -> dict:
    """
    Grades a single B2 English Writing Artile using the evolved playbook.

    Args:
        article_title (str): Student article title.
        article_text (str): Full article body.

    Returns:
        dict: {"scores": {TA: 3.0, ...}, "justification": "reasoning text"}
    """

    # Task prompt from rubric PDF (attached). This is fixed for all articles.
    TASK_PROMPT = """You have just seen the following advertisement in the university magazine:
Share with us what you think about the importance of physically attending classes in-person for university students instead of online classes. We're looking for articles about the benefits of studying in a classroom with a teacher. The best article will be published in our university magazine and the winner will receive a €200 gift card.
You have decided to contribute, Write an ARTICLE in which you:
• explain why physically attending classes can improve learning.
• mention why attending class at a university also provides the student with other resources and facilities.
• suggest ways in which teachers can encourage students to go to class.
Give your article a title. Write your ARTICLE in 180-220 words."""

    # Rubric from attached PDF - extract or hardcore (use extract_rubric from parse_data.py if dynamic).
    # Full prompt = task + article + rubric + playbook + instructions.
    prompt = f"""TASK: {TASK_PROMPT}
    
STUDENT ARTICLE TITLE: {article_title}
STUDENT ARTICLE:
{article_text}

RUBRIC:
{RUBRIC}

INSTRUCTIONS: Grade this article using the rubric. Analyze step-by-step. Output raw JSON: {{"reasoning": "analysis", "final_answer": "RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0"}}

**PLAYBOOK STRATEGIES:**
{PLAYBOOK_PROMPT}
"""
    
    response = generator.generate(question=prompt, context="", playbook=latest_playbook)

    try:
        justification = response.reasoning
        final_answer = response.final_answer
    except json.JSONDecodeError:
        raise ValueError("LLM output not valid JSON") 

    # Parse scores from final_answer.
    match = re.search(r'RESULTS\s*,\s*TA:(\d\.\d)\s*,\s*CC:(\d\.\d)\s*,\s*GR:(\d\.\d)\s*,\s*LR:(\d\.\d)\s*,\s*OWP:(\d\.\d)', final_answer, re.IGNORECASE)
    if match:
        scores = {
            "TA": float(match.group(1)),
            "CC": float(match.group(2)),
            "GR": float(match.group(3)),
            "LR": float(match.group(4)),
            "OWP": float(match.group(5)),
        }
    else:
        raise ValueError("Invalid score format in LLM output.")

    # Log for audit.
    log_entry = {"timestamp": datetime.now().isoformat(), "article_title": article_title, "scores": scores, "justification": justification}
    with open("logs/grades.log", "w") as f:
        json.dump(log_entry, f)
        f.write("\n")

    return {"scores": scores, "justification": justification}

# Example usage.
if __name__ == "__main__":
    # Test with attached example from PDF.
    title = "Being at classroom"
    text = "The goal of this article is to emphasize the importance of attending classes physically. With the COVID-19 CRISIS arrival, Every academic exam, task or class had to be done online. The use of platforms like Microsoft Teams and Zoom from teachers increased massively, remote classes have a huge advantage ahead physical in terms of availability and time-line dedication. However, online classes are less effective for improve learning. In my opinion, is way better attending a lesson in a calsroom, where you're face-to-face with your teacher, and your doubts can be solved. Besides, it's harder to see on camera when your proffesor writes some thing on the board. In conclusion, we need to make sure the advantages of online classes don't Persuade us about attending physically. Moreover, teachers have to atract students somehow. Maybe giving Some gifts or rewarding them on other way."

    result = grade_article(title, text)
    print(json.dumps(result, indent=2))