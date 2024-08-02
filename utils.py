import fitz
import os
from tqdm.auto import tqdm
import torch

def extract_text_from_pdf(directory: str)-> list[dict]:
  """
  Extracts text from all PDF files in a directory.
  """
  resumes = []
  for filename in os.listdir(directory):
      if filename.endswith(".pdf"):
          file_path = os.path.join(directory, filename)
          pdf_document = fitz.open(file_path)
          text = ""
          for page_num, page in tqdm(enumerate(pdf_document)):
              text += page.get_text()
          resumes.append({"filename": filename,
                            "page_count": page_num + 1,
                            "page_char_count": len(text),
                            "page_word_count": len(text.split(" ")),
                            "page_sentence_count_raw": len(text.split(". ")),
                            "page_token_count": len(text) / 4,  # 1 token
                            "text": text})
          pdf_document.close()
  return resumes