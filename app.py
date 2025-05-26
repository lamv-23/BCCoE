import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ——————————————————————————————
# Streamlit setup
# ——————————————————————————————
st.set_page_config(page_title="CBA Guide Assistant")
st.title("CBA Guide Assistant")

# load API key
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OpenAI key")
    st.stop()

# load & index PDFs once
if "vectorstore" not in st.session_state:
    text = ""
    for fn in os.listdir("data"):
        if fn.lower().endswith(".pdf"):
            with open(os.path.join("data", fn), "rb") as f:
                reader = PdfReader(f)
                for p in reader.pages:
                    text += (p.extract_text() or "") + "\n"
    chunks = CharacterTextSplitter(1000, 200).split_text(text)
    embeds = OpenAIEmbeddings(openai_api_key=api_key)
    st.session_state.vectorstore = DocArrayInMemorySearch.from_texts(chunks, embedding=embeds)

# prepare memory & chain
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    temperature=0.2,
    openai_api_key=api_key
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=st.session_state.vectorstore.as_retriever(),
    memory=memory,
    verbose=True
)

# render history
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# new user input
user_msg = st.chat_input("Ask about the CBA guides…")
if user_msg:
    # echo user
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # get answer
    with st.spinner("Thinking…"):
        result = conv_chain({"question": user_msg})
        answer = result["answer"]

    # display & store
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
