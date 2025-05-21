import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Streamlit page configuration
st.set_page_config(page_title="BCCoE Chatbot")
st.title("Chat with project PDFs")

# Retrieve OpenAI API key
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

# Load and extract text from all PDFs in the data/ folder
PDF_DIR = "data"
text = ""
for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith(".pdf"):
        path = os.path.join(PDF_DIR, filename)
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"

# Split the combined text into manageable chunks
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = splitter.split_text(text)

# Create embeddings and an in-memory vector store
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings)

# Define a custom system prompt to control responses
CUSTOM_SYSTEM_PROMPT = '''You are a helpful assistant that answers questions based only on the content of the provided PDFs.
Do not guess or make up answers. If you cannot find the answer, say "I'm not sure based on the provided documents."'''

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{CUSTOM_SYSTEM_PROMPT}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}"
)

# User input
query = st.text_input("Ask a question about the project documents:")
if query:
    with st.spinner("Thinking..."):
        try:
            # Retrieve relevant document chunks
            docs = vectorstore.similarity_search(query)

            # Instantiate the LLM with your desired creativity settings
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.3,          # Adjust creativity: 0.0 = factual, 1.0 = very creative
                top_p=0.9,                # Nucleus sampling parameter
                frequency_penalty=0.0,    # Penalise repeated tokens
                presence_penalty=0.0,     # Encourage/discourage new topics
                openai_api_key=api_key
            )

            # Build and run the QA chain
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            answer = chain.run(input_documents=docs, question=query)

            # Display the answer
            st.write("### Answer")
            st.write(answer)

        except Exception as err:
            st.error(f"Error generating answer: {err}")
