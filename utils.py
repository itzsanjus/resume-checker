import fitz
import os
from tqdm.auto import tqdm # for progress bar
import yake
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import language_tool_python

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available


def extract_text_from_all_pdfs(directory: str)-> list[dict]:
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

def extract_keywords(text : str, numOfKeywords = 100,language = "en",max_ngram_size = 3,deduplication_threshold = 0.9) -> list:
    """
    Extracts keywords from a given text using YAKE.
    """
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    return keywords


def match_resumes_to_keywords(resumes, keywords, top_k):
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    model.to("cuda") # requires a GPU installed
    # Create embeddings for the keywords
    keyword_embeddings = model.encode(keywords)

    ranked_resumes = []
    for resume in tqdm(resumes):
        # Create embedding for the resume text
        resume_embedding = model.encode(resume['text'])
        total_score = 0

        for keyword_embedding in keyword_embeddings:
            # Calculate the cosine similarity
            score = cosine_similarity([keyword_embedding], [resume_embedding]).flatten()[0]
            total_score += score

        # Normalize the score
        normalized_score = total_score / len(keywords)

        ranked_resumes.append((resume, normalized_score))

    # Sort the resumes by the normalized cosine similarity score in descending order
    ranked_resumes.sort(key=lambda x: x[1], reverse=True)

    # Print normalized scores and return the top_k resumes
    for resume, score in ranked_resumes:
        print(f"Resume: {resume['text']}\nNormalized Score: {score:.4f}\n")

    return [resume for resume, score in ranked_resumes[:top_k]]

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "google/gemma-2b-it"

# 1. Create quantization config for smaller model loading.
# For models that require 4-bit quantization (use this if low GPU memory is available)
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)

if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
  attn_implementation = "flash_attention_2"
else:
  attn_implementation = "sdpa"
print(f"[INFO] Using attention implementation: {attn_implementation}")

# 2. Pick a model we'd like to use
model_id = model_id # (we already set this above)
print(f"[INFO] Using model_id: {model_id}")

# 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

# 4. Instantiate the model
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                 torch_dtype=torch.float16, # datatype to use, we want float16
                                                 quantization_config=quantization_config if use_quantization_config else None,
                                                 low_cpu_mem_usage=False, # use full memory
                                                 attn_implementation=attn_implementation) # which attention version to use

if not use_quantization_config: # quantization takes care of device setting automatically, so if it's not used, send model to GPU
    llm_model.to("cuda")

def prompt_formatter(context_items) -> str:
    """
    Augments query with text-based context from context_items.

    """
    # Create a base prompt with examples to help the model
    base_prompt = "Summarize the following resume:\n"
    base_prompt += context_items['text']

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt


def summarize_resume(resume):
    text = resume['text']
    prompt = prompt_formatter(text)
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                temperature=0.7, # lower temperature = more deterministic outputs, higher temperature = more creative outputs
                                do_sample=True,
                                max_new_tokens=512) # how many new tokens to generate from prompt

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])
    return output_text



def analyze_resume(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    feedback = []
    for match in matches:
        feedback.append({
            "error": match.message,
            "suggestion": match.replacements,
            "context": match.context,
        })
    return feedback
