import os
import json
import streamlit as st
import pandas as pd
from opik import track
from grade_article import grade_article

st.title("UPV CEFR B2 Article Grader")

# Upload or input.
article_title = st.text_input("Article Title")
article_text = st.text_area("Article Text", height=300)

@track(project_name=os.getenv("OPIK_PROJECT_NAME"))
def ui_grade(title, text):
    return grade_article(title, text)

if st.button("Grade Article"):
    if article_title and article_text:
        with st.spinner("Grading..."):
            result = ui_grade(article_title, article_text)
        
        st.success("Grading Complete!")
        st.json(result["scores"])
        st.markdown("### Justification")
        st.write(result["justification"])
    else:
        st.error("Provide title and text.")

# Upload student articles in batch (CSV with title, text columns).
uploaded_file = st.file_uploader("Batch CSV Upload", type="csv")
if uploaded_file:
    try:
        
        # Reads CSV files with semicolon separation, handles quotes, and uses python engine for multi-line.
        df = pd.read_csv(uploaded_file, sep=";", quotechar='"', engine='python')

        # Validate columns.
        if 'title' not in df.columns or 'text' not in df.columns:
            st.error("CSV must have 'title' and 'text' columns.")
        else:
            grades = []
            for _, row in df.iterrows():
                grades.append(grade_article(row["title"], row["text"]))
            st.download_button("Download Grades", data=json.dumps(grades, indent=2), file_name="grades.json")
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}. Ensure semicolon-separated with quoted multi-line text.")

# TO RUN: streamlit run grader_ui.py
# TO STOP: `ctrl + c` inside terminal.