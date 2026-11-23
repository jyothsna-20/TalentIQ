import gradio as gr
import pandas as pd
import numpy as np
import re
import ast
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_resume_text(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

skills_list = [
    "python", "java", "c++", "c", "sql", "javascript", "html", "css",
    "react", "node.js", "angular", "mongodb", "mysql", "postgresql",
    "machine learning", "deep learning", "artificial intelligence",
    "data science", "data analysis", "pandas", "numpy", "matplotlib",
    "scikit-learn", "tensorflow", "pytorch", "nlp", "natural language processing",
    "computer vision", "excel", "power bi", "tableau", "aws", "azure",
    "google cloud", "docker", "git", "github", "linux", "flask",
    "django", "fastapi", "streamlit", "gradio", "statistics",
    "communication", "leadership", "problem solving", "project management",
    "teamwork", "data visualization", "cloud computing", "api",
    "rest api", "kubernetes", "transformers", "prompt engineering"
]

def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in skills_list:
        pattern = r'(?<!\w)' + re.escape(skill.lower()) + r'(?!\w)'

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills

def match_resumes(resumes, job_description):

    documents = resumes + [job_description]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    job_vector = tfidf_matrix[-1]
    resume_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(resume_vectors, job_vector).flatten()

    results = []

    for i, score in enumerate(similarities):
        resume_skills = extract_skills(resumes[i])
        job_skills = extract_skills(job_description)

        matched_skills = list(set(resume_skills) & set(job_skills))
        missing_skills = list(set(job_skills) - set(resume_skills))

        results.append({
            "Candidate": f"Candidate {i+1}",
            "Match Score (%)": round(score * 100, 2),
            "Matched Skills": matched_skills,
            "Missing Skills": missing_skills,
            "Matched Skill Count": len(matched_skills)
        })

    results = sorted(
        results,
        key=lambda x: x["Match Score (%)"],
        reverse=True
    )

    return results

def process_resumes(files, job_description):

    if not files:
        return pd.DataFrame()

    if not job_description.strip():
        return pd.DataFrame()

    resumes = []
    candidate_names = []

    for file in files:
        text = extract_resume_text(file)
        resumes.append(text)
        candidate_names.append(file.split("/")[-1])

    results = match_resumes(resumes, job_description)

    for result in results:

        original_index = int(
            result["Candidate"].split()[-1]
        ) - 1

        result["Candidate"] = candidate_names[original_index]

        result["Skill Coverage (%)"] = calculate_skill_coverage(
            resumes[original_index],
            job_description
        )

    for rank, result in enumerate(results, start=1):
        result["Rank"] = rank

    df = pd.DataFrame(results)

    df = df[
        [
            "Rank",
            "Candidate",
            "Match Score (%)",
            "Skill Coverage (%)",
            "Matched Skills",
            "Missing Skills",
            "Matched Skill Count"
        ]
    ]

    return df

skill_categories = {

    "Programming Languages": [
        "python", "java", "c++", "c", "javascript"
    ],

    "Data Science & Machine Learning": [
        "machine learning", "deep learning",
        "artificial intelligence", "data science",
        "data analysis", "pandas", "numpy",
        "matplotlib", "scikit-learn",
        "tensorflow", "pytorch", "statistics",
        "data visualization", "nlp",
        "natural language processing",
        "computer vision", "transformers",
        "prompt engineering"
    ],

    "Web Development": [
        "html", "css", "react", "node.js",
        "angular", "flask", "django",
        "fastapi", "streamlit", "gradio"
    ],

    "Databases": [
        "sql", "mysql", "mongodb", "postgresql"
    ],

    "Cloud & DevOps": [
        "aws", "azure", "google cloud",
        "docker", "kubernetes", "linux"
    ],

    "Tools & Technologies": [
        "git", "github", "api", "rest api", "excel",
        "power bi", "tableau"
    ],

    "Professional Skills": [
        "communication", "leadership",
        "problem solving", "project management",
        "teamwork"
    ]
}

def extract_categorized_skills(text):

    text = text.lower()
    extracted = {}

    for category, skills in skill_categories.items():

        found = []

        for skill in skills:
            pattern = r'(?<!\w)' + re.escape(skill.lower()) + r'(?!\w)'

            if re.search(pattern, text):
                found.append(skill)

        if found:
            extracted[category] = found

    return extracted

def compare_categorized_skills(resume_text, job_text):

    resume_skills = extract_categorized_skills(resume_text)
    job_skills = extract_categorized_skills(job_text)

    comparison = {}

    for category, job_skill_list in job_skills.items():

        resume_skill_list = resume_skills.get(category, [])

        matched = [
            skill for skill in job_skill_list
            if skill in resume_skill_list
        ]

        missing = [
            skill for skill in job_skill_list
            if skill not in resume_skill_list
        ]

        comparison[category] = {
            "Matched Skills": matched,
            "Missing Skills": missing
        }

    return comparison

def calculate_skill_coverage(resume_text, job_text):

    comparison = compare_categorized_skills(
        resume_text,
        job_text
    )

    total_required_skills = 0
    total_matched_skills = 0

    for category, result in comparison.items():

        total_required_skills += len(result["Matched Skills"])
        total_required_skills += len(result["Missing Skills"])

        total_matched_skills += len(result["Matched Skills"])

    if total_required_skills == 0:
        return 0

    coverage = (
        total_matched_skills / total_required_skills
    ) * 100

    return round(coverage, 2)

import gradio as gr
import html
import ast


# =========================================================
# 1. SKILL FORMATTER
# =========================================================

def format_skills(value):

    if value is None:
        return []

    if isinstance(value, list):
        skills = value

    else:
        value = str(value).strip()

        if value.lower() in ["", "nan", "none", "[]"]:
            return []

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                skills = parsed
            else:
                skills = value.split(",")

        except:
            skills = value.split(",")

    cleaned_skills = []

    for skill in skills:

        skill = str(skill).strip().strip("'\"")

        if skill and skill.lower() not in ["nan", "none"]:
            cleaned_skills.append(skill)

    return cleaned_skills


# =========================================================
# 2. CANDIDATE CARD RENDERER
# =========================================================

def render_candidate_dashboard(results):

    if results is None or results.empty:

        return """
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <h3>No candidates available</h3>
            <p>Upload resumes and run an analysis to get started.</p>
        </div>
        """

    dashboard = ""

    for index, row in results.iterrows():

        rank = row.get("Rank", index + 1)

        candidate = html.escape(
            str(row.get("Candidate", "Unknown"))
        )

        match_score = float(
            row.get("Match Score (%)", 0)
        )

        coverage = float(
            row.get("Skill Coverage (%)", 0)
        )

        skill_count = row.get(
            "Matched Skill Count", 0
        )

        matched_skills = format_skills(
            row.get("Matched Skills", "")
        )

        missing_skills = format_skills(
            row.get("Missing Skills", "")
        )

        matched_tags = "".join(
            f'<span class="skill-tag matched">{html.escape(skill)}</span>'
            for skill in matched_skills
        )

        missing_tags = "".join(
            f'<span class="skill-tag missing">{html.escape(skill)}</span>'
            for skill in missing_skills
        )

        top_candidate = str(rank) == "1"

        top_badge = (
            '<span class="top-badge">✦ TOP MATCH</span>'
            if top_candidate else ""
        )

        card_class = (
            "candidate-card top-card"
            if top_candidate
            else "candidate-card"
        )

        matched_content = (
            matched_tags
            if matched_skills
            else '<span class="muted">No matched skills detected</span>'
        )

        missing_content = (
            missing_tags
            if missing_skills
            else '<span class="success-text">No missing skills identified</span>'
        )

        dashboard += f"""

        <div class="{card_class}">

            <div class="candidate-header">

                <div class="candidate-identity">

                    <div class="rank-badge">
                        #{html.escape(str(rank))}
                    </div>

                    <div class="candidate-name">
                        {candidate}
                        {top_badge}
                    </div>

                </div>

                <div class="score-box">

                    <div class="score-label">
                        MATCH SCORE
                    </div>

                    <div class="score-value">
                        {match_score:.2f}%
                    </div>

                </div>

            </div>

            <div class="metrics-row">

                <div class="metric-block">

                    <div class="metric-label">
                        Skill Coverage
                    </div>

                    <div class="progress-track">

                        <div class="progress-fill"
                             style="width:{max(0,min(coverage,100))}%">
                        </div>

                    </div>

                    <div class="metric-value">
                        {coverage:.0f}%
                    </div>

                </div>

                <div class="skills-count">

                    <div class="metric-label">
                        Skills Matched
                    </div>

                    <div class="metric-value">
                        {html.escape(str(skill_count))}
                    </div>

                </div>

            </div>

            <div class="skills-section">

                <div class="skills-heading">
                    <span class="skills-icon">✓</span>
                    Matched Skills
                </div>

                <div class="skill-tags">
                    {matched_content}
                </div>

            </div>

            <div class="skills-section">

                <div class="skills-heading">
                    <span class="missing-icon">−</span>
                    Missing Skills
                </div>

                <div class="skill-tags">
                    {missing_content}
                </div>

            </div>

        </div>

        """

    return dashboard


# =========================================================
# 3. ANALYSIS DISPLAY CALLBACK WITH CSV EXPORT
# =========================================================

def run_analysis_display(files, job_description):

    summary, results = analyze_and_summarize(
        files,
        job_description
    )

    if results is None or results.empty:

        return summary, """
        <div class="empty-state">
            <h3>No candidate results</h3>
            <p>Please check your input and try again.</p>
        </div>
        """, None

    candidate_dashboard = render_candidate_dashboard(results)

    csv_path = None

    try:
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        )

        csv_path = temp_file.name
        temp_file.close()

        results.to_csv(
            csv_path,
            index=False
        )

    except Exception as e:

        print("CSV export error:", e)
        csv_path = None

    return summary, candidate_dashboard, csv_path


# =========================================================
# 4. CUSTOM DESIGN
# =========================================================

custom_css = """

/* GLOBAL */

body {
    background: #f4f7fb !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
    font-family: Inter, sans-serif !important;
}


/* HERO HEADER */

#main-header {
    background: linear-gradient(135deg, #0b1220, #172c46);
    padding: 42px;
    border-radius: 24px;
    color: white;
    margin-bottom: 30px;
    border: 1px solid #233b57;
    box-shadow: 0 12px 35px rgba(15,23,42,0.10);
}

#main-header h1 {
    color: white !important;
    font-size: 38px !important;
    font-weight: 750 !important;
    margin-bottom: 8px !important;
    letter-spacing: -1px;
}

#main-header p {
    color: #cbd5e1 !important;
    font-size: 15px !important;
}

#hero-tag {
    display: inline-block;
    color: #5eead4;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 15px;
}

#hero-description {
    color: #94a3b8;
    line-height: 1.8;
    max-width: 800px;
}


/* INPUT AREAS */

textarea {
    border-radius: 14px !important;
}

input {
    border-radius: 12px !important;
}


/* ANALYZE BUTTON */

#analyze-btn {
    background: linear-gradient(135deg, #0f766e, #0d9488) !important;
    color: white !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    height: 55px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(15,118,110,0.15);
    transition: 0.2s ease;
}

#analyze-btn:hover {
    background: #115e59 !important;
}


/* CANDIDATE CARDS */

.candidate-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 25px;
    margin: 18px 0;
    box-shadow: 0 4px 14px rgba(15,23,42,0.04);
}

.top-card {
    border: 1px solid #99d8ce;
    box-shadow: 0 5px 20px rgba(15,118,110,0.08);
}


/* CANDIDATE HEADER */

.candidate-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 20px;
}

.candidate-identity {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
}

.rank-badge {
    width: 44px;
    height: 44px;
    background: #0f172a;
    color: white;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 750;
    font-size: 15px;
}

.candidate-name {
    font-size: 17px;
    font-weight: 700;
    color: #0f172a;
    word-break: break-word;
}

.top-badge {
    display: inline-block;
    margin-left: 10px;
    background: #ccfbf1;
    color: #0f766e;
    padding: 6px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 0.5px;
}


/* MATCH SCORE */

.score-box {
    text-align: right;
}

.score-label {
    font-size: 11px;
    color: #64748b;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.score-value {
    color: #0f766e;
    font-size: 29px;
    font-weight: 800;
    margin-top: 4px;
}


/* METRICS */

.metrics-row {
    display: flex;
    align-items: center;
    gap: 35px;
    flex-wrap: wrap;
    padding: 18px 0;
    border-top: 1px solid #f1f5f9;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 20px;
}

.metric-block {
    flex: 1;
    min-width: 200px;
}

.metric-label {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 16px;
    color: #0f172a;
    font-weight: 750;
}

.progress-track {
    height: 8px;
    background: #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    margin: 10px 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #0d9488, #2dd4bf);
    border-radius: 10px;
}

.skills-count {
    min-width: 130px;
}


/* SKILLS */

.skills-section {
    margin-top: 20px;
}

.skills-heading {
    font-size: 13px;
    font-weight: 700;
    color: #334155;
    margin-bottom: 12px;
}

.skills-icon {
    color: #0f766e;
    margin-right: 8px;
}

.missing-icon {
    color: #d97706;
    margin-right: 8px;
}

.skill-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.skill-tag {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 9px;
    font-size: 12px;
    margin: 2px;
    word-break: break-word;
}

.matched {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #d1fae5;
}

.missing {
    background: #fff7ed;
    color: #c2410c;
    border: 1px solid #ffedd5;
}

.muted {
    color: #94a3b8;
    font-size: 13px;
}

.success-text {
    color: #0f766e;
    font-size: 13px;
    background: #ecfdf5;
    padding: 9px 12px;
    border-radius: 9px;
}


/* EMPTY STATE */

.empty-state {
    background: white;
    border: 1px solid #e2e8f0;
    padding: 35px;
    border-radius: 18px;
    color: #64748b;
    text-align: center;
}

.empty-state h3 {
    color: #0f172a;
}

.empty-icon {
    font-size: 30px;
    color: #0f766e;
}


/* FOOTER */

#app-footer {
    text-align: center;
    color: #64748b;
    padding: 25px 10px;
    border-top: 1px solid #e2e8f0;
    margin-top: 30px;
    font-size: 13px;
}

footer {
    display: none !important;
}
"""


# =========================================================
# 5. BUILD WEBSITE
# =========================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="TalentIQ | AI Resume Matcher"
) as demo:


    # HERO

    gr.HTML("""
    <div id="main-header">

        <div id="hero-tag">
            AI TALENT INTELLIGENCE
        </div>

        <h1>✦ TalentIQ</h1>

        <p>
        Intelligent Resume Screening
        </p>

        <div id="hero-description">
        Analyse resumes against job descriptions, discover skill gaps
        and rank candidates using Natural Language Processing
        and machine learning techniques.
        </div>

    </div>
    """)


    # INPUTS

    with gr.Row():

        with gr.Column(scale=6):

            gr.Markdown("### 📋 Job Description")

            job_description = gr.Textbox(
                label="",
                placeholder="Paste the complete job description here...",
                lines=12
            )


        with gr.Column(scale=5):

            gr.Markdown("### 📄 Candidate Resumes")

            resume_upload = gr.File(
                label="Upload Resumes (PDF)",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath"
            )

            gr.Markdown(
                "Upload multiple PDF resumes to compare candidates."
            )


    # BUTTON

    analyze_button = gr.Button(
        "✦  Analyze & Rank Candidates",
        elem_id="analyze-btn"
    )


    gr.Markdown("---")


    # SUMMARY

    gr.Markdown("## 📊 Screening Insights")

    summary_output = gr.HTML(
        value="""
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <h3>Ready to analyse candidates</h3>
            <p>Upload resumes and enter a job description to begin.</p>
        </div>
        """
    )


    # RANKINGS

    gr.Markdown("## 🏆 Candidate Rankings")

    candidate_output = gr.HTML(
        value="""
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <h3>Candidate rankings will appear here</h3>
            <p>Your screening results will be displayed after analysis.</p>
        </div>
        """
    )

    # CSV DOWNLOAD

    csv_download = gr.File(
        label="📥 Download Candidate Results (CSV)"
    )


    # FOOTER

    gr.HTML("""
    <div id="app-footer">
        ✦ TalentIQ · Intelligent Resume Screening
        <br><br>
        <span style="font-size:11px;">
        Powered by Python · NLP · TF-IDF · Cosine Similarity · Skill Extraction
        </span>
    </div>
    """)


    # CALLBACK

    analyze_button.click(
        fn=run_analysis_display,
        inputs=[resume_upload, job_description],
        outputs=[summary_output, candidate_output, csv_download]
    )


# =========================================================
# 6. LAUNCH
# =========================================================

def create_summary(results):

    if results.empty:
        return "No candidates analyzed."

    total_candidates = len(results)

    top_candidate = results.iloc[0]["Candidate"]

    highest_match = results.iloc[0]["Match Score (%)"]

    avg_match = round(
        results["Match Score (%)"].astype(float).mean(), 2
    )

    summary_html = f"""
    <div style="
        display:flex;
        flex-wrap:wrap;
        gap:16px;
        margin:20px 0;
    ">

        <div style="
            flex:1;
            min-width:180px;
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:16px;
            padding:22px;
        ">
            <div style="color:#64748b;font-size:13px;">
                TOTAL RESUMES
            </div>
            <div style="
                font-size:30px;
                font-weight:700;
                color:#0f172a;
                margin-top:10px;
            ">
                {total_candidates}
            </div>
        </div>

        <div style="
            flex:1;
            min-width:180px;
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:16px;
            padding:22px;
        ">
            <div style="color:#64748b;font-size:13px;">
                HIGHEST MATCH
            </div>
            <div style="
                font-size:30px;
                font-weight:700;
                color:#0f766e;
                margin-top:10px;
            ">
                {highest_match}%
            </div>
        </div>

        <div style="
            flex:1;
            min-width:180px;
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:16px;
            padding:22px;
        ">
            <div style="color:#64748b;font-size:13px;">
                AVG. MATCH SCORE
            </div>
            <div style="
                font-size:30px;
                font-weight:700;
                color:#0f172a;
                margin-top:10px;
            ">
                {avg_match}%
            </div>
        </div>

        <div style="
            flex:1;
            min-width:220px;
            background:#0f172a;
            border:1px solid #0f172a;
            border-radius:16px;
            padding:22px;
        ">
            <div style="color:#94a3b8;font-size:13px;">
                TOP CANDIDATE
            </div>
            <div style="
                font-size:17px;
                font-weight:600;
                color:#ffffff;
                margin-top:10px;
                word-break:break-word;
            ">
                {top_candidate}
            </div>
        </div>

    </div>
    """

    return summary_html

def analyze_and_summarize(files, job_description):

    if not job_description or not job_description.strip():
        empty_results = pd.DataFrame()
        return "⚠️ Please enter a job description before analyzing resumes.", empty_results

    if not files:
        empty_results = pd.DataFrame()
        return "⚠️ Please upload at least one resume.", empty_results

    results = process_resumes(files, job_description)

    if results.empty:
        return "No candidates analyzed.", results

    summary = create_summary(results)

    return summary, results

import tempfile
import os

def run_analysis_display(files, job_description):

    summary, results = analyze_and_summarize(
        files,
        job_description
    )

    if results is None or results.empty:

        return summary, """
        <div class="empty-state">
            <h3>No candidate results</h3>
            <p>Please check your input and try again.</p>
        </div>
        """, None

    candidate_dashboard = render_candidate_dashboard(results)

    csv_path = None

    if not results.empty:

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        )

        csv_path = temp_file.name
        temp_file.close()

        results.to_csv(
            csv_path,
            index=False
        )

    return summary, candidate_dashboard, csv_path

# Launch the app on Render
import os
demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 10000)))
