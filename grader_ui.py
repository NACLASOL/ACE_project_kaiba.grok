import streamlit as st
import pandas as pd
from grade_article import *

st.title("UPV CEFR B2 Article Grader")

# Upload or input.
article_title = st.text_input("Article Title")
article_text = st.text_area("Article Text", height=300)

if st.button("Grade Article"):
    if article_title and article_text:
        with st.spinner("Grading..."):
            result = grade_article(article_title, article_text)
        
        st.success("Grading Complete!")
        st.json(result["scores"])
        st.markdown("### Justification")
        st.write(result["justification"])
    else:
        st.error("Provide title and text.")

# Upload student articles in batch (CSV with title, text columns).
uploaded_file = st.file_uploader("Batch CSV Upload", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    grades = []
    for _, row in df.iterrows():
        grades.append(grade_article(row["title"], row["text"]))
        st.download_button("Download Grades", data=json.dumps(grades, indent=2), file_name="grades.json")

# TO RUN: streamlit run grader_ui.py