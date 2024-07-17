from flask import Flask, request, render_template
from utils import extract_text_from_pdf, extract_keywords, match_resumes_to_keywords, summarize_resume, analyze_resume

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    resume_files = request.files.getlist('resumes')
    job_description = request.form['job_description']
    
    resumes = []
    for resume_file in resume_files:
        resume_text = extract_text_from_pdf(resume_file)
        resumes.append({"text": resume_text, "filename": resume_file.filename})

    keywords = extract_keywords(job_description)
    matched_resumes = match_resumes_to_keywords(resumes, keywords)
    
    results = []
    for resume in matched_resumes:
        summary = summarize_resume(resume['text'])
        feedback = analyze_resume(resume['text'])
        results.append({
            "filename": resume['filename'],
            "summary": summary,
            "feedback": feedback
        })

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
