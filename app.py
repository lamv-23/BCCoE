import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ——————————————————————————————
# 📄 Page config & session setup
# ——————————————————————————————
st.set_page_config(page_title="BCCoE Chatbot")
st.title("Chat with project PDFs")

# initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ——————————————————————————————
# 🔑 Load API key & vector store once
# ——————————————————————————————
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

if "vectorstore" not in st.session_state:
    # load and extract all PDF text
    text = ""
    for fn in os.listdir("data"):
        if fn.lower().endswith(".pdf"):
            with open(os.path.join("data", fn), "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        text += txt + "\n"

    # split into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # embed and store in‐memory
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    st.session_state.vectorstore = DocArrayInMemorySearch.from_texts(
        chunks, embedding=embeddings
    )

# ——————————————————————————————
# 🧠 Define custom prompt for personality & follow‐up
# ——————————————————————————————
CUSTOM_SYSTEM_PROMPT = '''You are a friendly, conversational assistant who speaks like a colleague over coffee.
Answer questions using only the provided PDFs. If you don’t know, say "I’m not sure based on these documents."
After your answer, ask one follow-up question to guide the user deeper.'''

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""{CUSTOM_SYSTEM_PROMPT}

Context:
{{context}}

Question:
{{question}}"""
)

# ——————————————————————————————
# 💬 Render chat history
# ——————————————————————————————
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ——————————————————————————————
# ✍️ New user input
# ——————————————————————————————
user_input = st.chat_input("Ask a question about the project documents:")
if user_input:
    # record & display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # generate assistant response
    with st.spinner("Thinking..."):
        docs = st.session_state.vectorstore.similarity_search(user_input)
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,       # mild creativity
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            openai_api_key=api_key
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        answer = chain.run(input_documents=docs, question=user_input)

    # record & display assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
