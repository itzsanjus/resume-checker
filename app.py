from flask import Flask, request, render_template
from flask import send_from_directory
import os
import shutil
from utils import extract_text_from_pdf, extract_keywords, match_resumes_to_keywords, summarize_resume, highlights, highlights_keywords, analyze_resume


app = Flask(__name__,template_folder='/content/drive/MyDrive/templates')

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
  resume_files = request.files.getlist('resumes')
  job_description = request.form['job_description']
  keywords = request.form.get('keywords')
  top_n = request.form.get('top_n')
  top_k = request.form.get('top_k')

  if not resume_files or (not job_description and not keywords) or not top_n:
    flash('Please upload at least one resume, provide a job description or keywords, and specify the number of top matched resumes.')
    return redirect(url_for('home'))
  top_n = int(top_n)
  if top_n < 1:
      flash('The number of top matched resumes must be at least 1')
      return redirect(url_for('home'))

  resume_directory = '/content/drive/MyDrive/resumes'
  if os.path.exists(resume_directory):
    shutil.rmtree(resume_directory)
  os.makedirs(resume_directory)
  print("Directory created successfully")


  for resume_file in resume_files:
    resume_path = os.path.join(resume_directory, resume_file.filename)
    resume_file.save(resume_path)  # Save the resume file
    print(f"Saved resume: {resume_path}")
   
  resumes = extract_text_from_pdf(resume_directory)
  print(f"Extracted {len(resumes)} resumes from PDF files.")

  if keywords:
    keywords_list = keywords.split(',')
  else:
    top_k = int(top_k)
    if top_k < 1:
      flash('The number of top extracted keywords be at least 1')
      return redirect(url_for('home'))

    keywords_list = extract_keywords(job_description,top_k)
    print(f"Extracted {len(keywords_list)} keywords from job description.")


  
  top_n = min(top_n, len(resumes))
  matched_resumes = match_resumes_to_keywords(resumes, keywords_list, top_n)
 
  results = []
  for resume in matched_resumes:
    output_text = summarize_resume(resume)
    print(f"Summary generated for {resume['filename']}")
    output_text = format_summary(output_text)
    if keywords:
      highlight = highlights_keywords(resume,keywords)
      print(f"Highlights generated for {resume['filename']}")
      highlight = format_summary(highlight)
    else:
      highlight = highlights(resume,job_description)
      print(f"Highlights generated for {resume['filename']}")
      highlight = format_summary(highlight)
    feedback = analyze_resume(resume['text'])
    results.append({
        "filename": resume['filename'],
        "summary": output_text,
        "highlights": highlight,
        "feedback": feedback
    })


  return render_template('result.html', results=results)
# Existing imports and app initialization

# Add this new route to serve the files
@app.route('/download/<filename>')
def download_file(filename):
  # Replace with the directory where resumes are saved/uploaded
  resume_directory = '/content/drive/MyDrive/resumes'
  return send_from_directory(resume_directory, filename)



if __name__ == '__main__':
  app.run()