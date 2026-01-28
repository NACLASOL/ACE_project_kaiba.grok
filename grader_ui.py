import os
import json
import logging
import streamlit as st
import pandas as pd
from opik import *
from datetime import datetime
from grade_article import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="UPV CEFR Article Grader",
    page_icon="✏️",
    layout="wide",
)

st.title("UPV CEFR B2 Article Grader")
st.markdown(
    """
This application grades English articles using CEFR B2 rubric standards with AI-powered analysis.
Each grade is traced and logged for quality assurance.
"""
)

# === VALIDATION HELPERS ===
def validate_result(result: dict) -> bool:
    """
    Validate that "grade_article()" returned expected struture.

    Args:
        result: Result from "grade_article()"
    
    Returns:
        bool: True if valid.

    Raises:
        ValueError: if structure invalid.
    """

    if not isinstance(result, dict):
        raise ValueError("Grade result must be a dictionary.")
    
    required_keys = ["scores", "justification", "metadata"]
    for key in required_keys:
        if key not in result:
            raise ValueError(f"Grade result missing required key: {key}")
        
    if not isinstance(result["scores"], dict):
        raise ValueError("Scores must be a dictionary.")
    
    required_scores = ["TA", "CC", "GR", "LR", "OWP"]
    for score_key in required_scores:
        if score_key not in result["scores"]:
            raise ValueError(f"Scores missing required key: {score_key}")
        
        score_val = result["scores"][score_key]
        if not isinstance(score_val, (int, float)):
            raise ValueError(f"Score {score_key} must be numeric.")
        if not (1.0 <= score_val <= 5.0):
            raise ValueError(f"Score {score_key}={score_val} outside range [1.0, 5.0]")
    
    return True

def validate_article_input(title: str, text: str) -> list:
    """
    Validate article inputs.

    Args:
        title: Article title.
        text: Article text.

    Returns:
        list: List of errors strings (empty if valid).
    """
    errors = []

    # Title validation.
    if not title or len(title.strip()) == 0:
        errors.append("Title cannot be empty.")
    elif len(title) > 200:
        errors.append("Title too long (max 200 characters).")

    # Text valdation.
    if not text or len(text.strip()) == 0:
        errors.append("Article text cannot be empty.")
    elif len(text) < 50:
        errors.append("Article too short (minimum 50 characters).")
    elif len(text) > 5000:
        errors.append("Article too long (maximum 5000 characters).")

    return errors

# === GRADING FUNCTIONS ===
@track(project_name=os.getenv("OPIK_PROJECT_NAME"))
def ui_grade(title: str, text: str):
    """
    Grade a single article via UI.

    Args:
        title: Article title from user input.
        text: Article text from user input.

    Returns:
        dict: {scores: {...}, justification: "...", metadata: {...}}

    Raises:
        ValueError: If inputs invalid or grading.
    """

    input_errors = validate_article_input(title, text)
    if input_errors:
        raise ValueError("; ".join(input_errors))
    

    # Validate result.
    # validate_result(result)

    return grade_article(title, text)

@track(project_name=os.getenv("OPIK_PROJECT_NAME"))
def ui_batch_grade(df: pd.DataFrame) -> list:
    """
    Grade multiple articles from batch CSV upload.

    Args:
        df: DataFrame with 'title' and 'text' columns.

    Returns:
        list: [{scores, justification, metadata}, ...] for each article.

    Raises:
        ValueError: If required columns missing.
    """

    # Validate columns.
    if 'title' not in df.columns or 'text' not in df.columns:
        raise ValueError(
            "CSV must have 'title' and 'text' columns."
            "Got columns: " + ", ".join(df.columns)
        )
    
    grades = []
    errors = []

    for idx, row in df.iterrows():
        try:
            # Each iteration: grade_article created child span automatically.
            result = grade_article(row["title"], row["text"])
            validate_result(result)
            grades.append(result)


        except Exception as e:
            error_info = {
                "row": int(idx),
                "title": str(row["title"])[:100],
                "error": str(e),
                "error_type": type(e).__name__,
            }
            errors.append(error_info)
            logger.error(
                f"Batch grading failed at row {idx}."
                f"('{row['title']}'): {str(e)}"
            )

    if errors:
        logger.warning(
            f"Batch processing: {len(errors)} errors out of {len(df)} articles"
        )
    
    return grades

# === SIDEBAR CONFIGURATION ===
with st.sidebar:
    st.header("⚙️ Configuration")

    mode = st.radio(
        "Grading Mode",
        ["Single Article", "Batch Upload"],
        help = "Choose between grading a single artile or uploading multiple."
    )

    st.markdown("---")

    st.markdown("""
**About CEFR B2:**
                - European Framework reference level for upper-intermediate English.
                - Test: Grammar, vocabulary, coherence, task achievement...
                - Scale: 1.0 (Inadequate) to 5.0 (Excellent)

"""
)
    
# === MAIN INTERFACE ===
if mode == "Single Article":
    st.header("Grade a Single Article")

    col1, col2, = st.columns([1, 1])

    with col1:
        article_title = st.text_input(
            "Article Title",
            placeholder="e.g., 'Why Online Classes Matter'",
            max_chars=200,
            help="Enter the student's article title."
        )
    
    with col2:
        st.markdown("") # Visual spacer.

    article_text = st.text_area(
        "Article Text",
        placeholder="Paste the complete article here (minimum 50 characters)...",
        height=300,
        max_chars=5000,
        help="Full article body to be graded."
    )

    if st.button("Grade Article", use_container_width=True):
        # Validate inputs.
        input_errors = validate_article_input(article_title, article_text)

        if input_errors:
            for error in input_errors:
                st.error(f"❌ {error}")

        else:
            with st.spinner("⏳ Grading your article..."):
                try:
                    result = ui_grade(article_title, article_text)

                    # Display success.
                    st.success("✅ Grading Complete!")

                    # Display scores.
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Task Achievement", result["scores"]["TA"])

                    with col2:
                        st.metric("Cohesion & Coherence", result["scores"]["CC"])

                    with col3:
                        st.metric("Grammar", result["scores"]["GR"])
                    
                    with col4:
                        st.metric("Vocabulary", result["scores"]["LR"])
                    
                    with col5:
                        st.metric("Overall", result["scores"]["OWP"])

                    
                    # Justification.
                    st.markdown("### 📋 Detailed Justification")
                    st.write(result["justification"])

                    # Technical details.
                    with st.expander("📊 Technical Details"):
                        st.json({
                            "article_length": result["metadata"]["article_length"],
                            "model": result["metadata"]["model"],
                            "cerfr_level": result["metadata"]["cefr_level"],
                            "timestamp": result["metadata"]["timestamp"],
                        })

                        # Download option.
                        st.download_button(
                            label="Download Results as JSON",
                            data=json.dumps(result, indent=2),
                            file_name=f"grade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                except ValueError as e:
                    st.error(f"❌ Validation Error: {str(e)}")
                    logger.error(f"Validation error for '{article_title}': {str(e)}")

                except Exception as e:
                    st.error(f"❌ Grading Error: {str(e)}")
                    logger.error(
                        f"Grading error: {type(e).__name__}: {str(e)}"
                    )
                
    else:
        st.header("Batch Grade Articles.")

        st.markdown(
            """
            Upload a CSV file with the following columns:

            - **title**: Article title.
            - **text**: Article body.

            **Format Requirements:**
            - Separator: ';' (semicolon).
            - Quote character: '"' (double quotes).
            - Encoding: UTF-8
            """
        )

        # File uploader.
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type="csv",
            help="CSV with 'title' and 'text' columns, semicolon-separated"
        )

        if uploaded_file:
            try:
                df = pd.read_csv(
                    uploaded_file,
                    sep=";",
                    quotechar='"',
                    engine='python'
                )

                st.markdown(f"📊 Loaded {len(df)} articles.")

                # Preview.
                with st.expander ("👀 Preview"):
                    st.dataframe(df.head(), use_container_width=True)

                if st.button("🎯 Grade All Articles", use_container_width=True):
                    with st.spinner(f"⏳ Grading {len(df)} articles..."):
                        try:
                            grades = ui_batch_grade(df)

                            if not grades:
                                st.error("❌ No articles could be graded successfully")

                            else:
                                st.success(
                                    f"✅ Graded {len(grades)}/{len(df)} articles successfully"
                                )

                                # Results table.
                                results_data = []

                                for idx, grade in enumerate(grades):
                                    results_data.append({
                                        "Index": idx,
                                        "TA": grade["scores"]["TA"],
                                        "CC": grade["scores"]["CC"],
                                        "GR": grade["scores"]["GR"],
                                        "LR": grade["scores"]["LR"],
                                        "OWP": grade["scores"]["OWP"]
                                    })
                                
                                results_df = pd.DataFrame(results_data)
                                st.dataframe(results_df, use_container_width=True)

                                # Statistics.
                                col1, col2, col3, col4, col5 = st.columns(5)

                                with col1:
                                    st.metric("Avg TA", f"{results_df['TA'].mean():.2f}")

                                with col2:
                                    st.metric("Avg CC", f"{results_df['CC'].mean():.2f}")

                                with col3:
                                    st.metric("Avg GR", f"{results_df['GR'].mean():.2f}")
                                
                                with col4:
                                    st.metric("Avg LR", f"{results_df['LR'].mean():.2f}")

                                with col5:
                                    st.metric("Avg OWP", f"{results_df['OWP'].mean():.2f}")
                                
                                # Download all results.
                                st.download_button(
                                    label = "📥 Download All Grades as JSON",
                                    data=json.dumps(grades, indent=2),
                                    file_name=f"batch_grades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                        
                        except ValueError as e:
                            st.error(f"❌ Validation Error: {str(e)}")
                            logger.error(f"Batch validation error: {str(e)}")

                        except Exception as e:
                            st.error(f"❌ Batch grading Error: {str(e)}")
                            logger.error(
                                f"Batch grading error: {type(e).__name__}: {str(e)}"
                            )

            except Exception as e:
                st.error(
                    f"❌ Error reading CSV: {str(e)}\n\n"
                    "Ensure file is:\n"
                    "- Semicolon-separated (;)\n"
                    "- UTF-8 encoded\n"
                    "- Has 'title' and 'text' columns"
                )
                logger.error(f"CSV parsing error: {str(e)}")

# === FOOTER ===
st.markdown("---")
st.markdown(
    """
    **How to Use:**

    1. **Single Mode**: Paste an article and click Grade.
    2. **Batch Mode**: Upload a CSV with title/text columns.

    **All grades are traced and logged for quality assurance.**

    For issues, check logs or contact support.
    """
)