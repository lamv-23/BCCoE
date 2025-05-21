import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate

# Set Streamlit page config
st.set_page_config(page_title="BCCoE Chatbot")
st.title("Chat with project PDFs")

# Load OpenAI API key
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

# Load all PDF text from the data folder
PDF_DIR = "data"
text = ""
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        with open(os.path.join(PDF_DIR, filename), "rb") as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"

# Split into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text)

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings(api_key=api_key)
vectorstore = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings)

# Define custom prompt
CUSTOM_SYSTEM_PROMPT = '''You are a helpful assistant that answers questions based only on the content of the provided PDFs.
Do not guess or make up answers. If you cannot find the answer, say "I'm not sure based on the provided documents."'''

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{CUSTOM_SYSTEM_PROMPT}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}"
)

# Input question
query = st.text_input("Ask a question about the project documents:")
if query:
    with st.spinner("Thinking..."):
        try:
            docs = vectorstore.similarity_search(query)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, api_key=api_key)

            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            response = chain.run(input_documents=docs, question=query)

            st.write("### Answer")
            st.write(response)
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
