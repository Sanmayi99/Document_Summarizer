import gradio as gr
from transformers import pipeline
from PyPDF2 import PdfReader
import docx

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def read_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def summarize_file(file):
    if file.name.endswith(".pdf"):
        text = read_pdf(file)
    elif file.name.endswith(".docx"):
        text = read_docx(file)
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    else:
        return "Unsupported file format."

    if not text.strip():
        return "The file is empty or could not extract text."

    # Limit to first 1000 tokens for summarization
    text = text[:3000]
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

iface = gr.Interface(
    fn=summarize_file,
    inputs=gr.File(file_types=[".pdf", ".docx", ".txt"]),
    outputs="text",
    title="AI Document Summarizer",
    description="Upload a .pdf, .docx, or .txt file to get a summary using BART model."
)

if __name__ == "__main__":
    iface.launch(share=True)
