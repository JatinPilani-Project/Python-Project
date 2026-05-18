import os
import re
import pandas as pd
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_pdf_text(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_docx_text(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

with open("data/job_description.txt", "r") as file:
    job_description = clean_text(file.read())

resume_folder = "resumes"
resume_data = []

for file in os.listdir(resume_folder):
    file_path = os.path.join(resume_folder, file)

    if file.endswith(".pdf"):
        text = extract_pdf_text(file_path)

    elif file.endswith(".docx"):
        text = extract_docx_text(file_path)

    else:
        continue

    cleaned_resume = clean_text(text)

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [job_description, cleaned_resume]
    )

    similarity = cosine_similarity(vectors)[0][1]

    status = "Shortlisted" if similarity > 0.30 else "Rejected"

    resume_data.append({
        "Resume": file,
        "Score": round(similarity * 100, 2),
        "Status": status
    })

df = pd.DataFrame(resume_data)

if not df.empty:
    df = df.sort_values(by="Score", ascending=False)

output_path = "outputs/ranked_candidates.csv"
df.to_csv(output_path, index=False)

print("\nResume Screening Completed Successfully!\n")
print(df)