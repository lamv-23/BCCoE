# build_index.py
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Make sure OPENAI_API_KEY is set in your environment
api_key = os.environ["OPENAI_API_KEY"]

# 1) Read & combine all PDFs
full_text = ""
for fn in os.listdir("data"):
    if fn.lower().endswith(".pdf"):
        with open(os.path.join("data", fn), "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                txt = page.extract_text() or ""
                full_text += txt + "\n"

# 2) Split into chunks
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = splitter.split_text(full_text)

# 3) Embed & build FAISS index
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
index = FAISS.from_texts(chunks, embeddings)

# 4) Persist to disk
persist_dir = "faiss_index"
os.makedirs(persist_dir, exist_ok=True)
index.save_local(persist_dir)

print(f"Index built and saved to ./{persist_dir}")
