import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ——————————————————————————————
# 📄 Page config & custom CSS
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")
st.markdown("""
    <style>
      /* shrink headings */
      h1, h2 { font-size: 1.25rem !important; }
      h3      { font-size: 1.1rem !important; }
      /* colour the input box */
      .st-TextInput>div>div>input {
        background-color: #f0f0f5 !important;
      }
    </style>
""", unsafe_allow_html=True)
st.title("BCCoE CBA Guide Assistant")

# Welcome banner
st.markdown("""
<div style="background-color:#333; color:white; padding:15px; border-radius:8px; margin-bottom:20px">
  <h4 style="margin:0">👋 Welcome to the CBA Guide Assistant</h4>
  <p style="margin:5px 0 0">Ask me anything about cost-benefit analysis guides.</p>
</div>
""", unsafe_allow_html=True)

# initialise chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ——————————————————————————————
# 🔑 Load API key & vectorstore once
# ——————————————————————————————
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

if "vectorstore" not in st.session_state:
    # load PDFs
    text = ""
    for fn in os.listdir("data"):
        if fn.lower().endswith(".pdf"):
            with open(os.path.join("data", fn), "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    text += txt + "\n"
    # chunk, embed & index
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
    chunks   = splitter.split_text(text)
    embeds   = OpenAIEmbeddings(openai_api_key=api_key)
    st.session_state.vectorstore = DocArrayInMemorySearch.from_texts(chunks, embedding=embeds)

# ——————————————————————————————
# 🧠 Build the conversational chain
# ——————————————————————————————
CUSTOM_SYSTEM_PROMPT = """You are a friendly, conversational assistant expert in cost–benefit analysis (CBA) guides.
Use only the guides’ content. Cite sections when you reference them.
Give clear, concise answers in markdown, with headings, lists or code blocks when helpful.
If you don’t know, say “I’m not sure based on the guides—please check the relevant section or contact a team member.”"""

# Memory to hold prior user/assistant turns
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    temperature=0.2,       # low randomness for consistency
    top_p=0.9,
    max_tokens=700,
    openai_api_key=api_key
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=st.session_state.vectorstore.as_retriever(),
    memory=memory,
    chain_type_kwargs={"prompt": PromptTemplate(
        input_variables=["system_prompt","question","chat_history","context"],
        template=(
            "{system_prompt}\n\n"
            "Context:\n{context}\n\n"
            "Conversation so far:\n{chat_history}\n\n"
            "User question:\n{question}"
        )
    )},
    system_prompt=CUSTOM_SYSTEM_PROMPT
)

# ——————————————————————————————
# 💬 Render chat history
# ——————————————————————————————
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# ——————————————————————————————
# ✍️ Handle a new user message
# ——————————————————————————————
user_msg = st.chat_input("Type your question here…")
if user_msg:
    # echo user
    st.session_state.chat_history.append({"role":"user","content":user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # get answer (this uses both retrieval and full convo history)
    with st.spinner("Thinking…"):
        result = conv_chain({"question": user_msg})
        answer = result["answer"]

    # display & store
    st.session_state.chat_history.append({"role":"assistant","content":answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
