import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Page config
st.set_page_config(page_title="BCCoE Chatbot")
st.title("Chat with project PDFs")

# API key
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

# Load PDFs from data/ folder
PDF_DIR = "data"
text = ""
for fn in os.listdir(PDF_DIR):
    if fn.lower().endswith(".pdf"):
        with open(os.path.join(PDF_DIR, fn), "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

# Split into chunks
splitter = CharacterTextSplitter(
    separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
)
chunks = splitter.split_text(text)

# Build embeddings + in-memory store
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings)

# System prompt
CUSTOM_SYSTEM_PROMPT = '''You are a helpful assistant that answers questions based only on the content of the provided PDFs.
Do not guess or make up answers. If you cannot find the answer, say "I'm not sure based on the provided documents."'''

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{CUSTOM_SYSTEM_PROMPT}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}"
)

# User query
query = st.text_input("Ask a question about the project documents:")
if query:
    with st.spinner("Thinking..."):
        try:
            docs = vectorstore.similarity_search(query)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=api_key)

            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            answer = chain.run(input_documents=docs, question=query)

            st.write("### Answer")
            st.write(answer)

        except Exception as err:
            st.error(f"Error generating answer: {err}")
