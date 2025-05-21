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
st.title("BCCoE Training Assistant")

# Welcome banner with darker background and white text
st.markdown(
    """
    <div style="background-color:#333333; color:white; padding:15px; border-radius:8px; margin-bottom:20px">
      <h4 style="margin:0">👋 Welcome to the BCCoE Training Assistant</h4>
      <p style="margin:5px 0 0">
        Ask me anything about developing business cases and I'll do my best to help.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    text = ""
    for fn in os.listdir("data"):
        if fn.lower().endswith(".pdf"):
            with open(os.path.join("data", fn), "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        text += txt + "\n"

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    st.session_state.vectorstore = DocArrayInMemorySearch.from_texts(
        chunks, embedding=embeddings
    )

# ——————————————————————————————
# 🧠 Define custom prompts for map-reduce flow
# ——————————————————————————————
CUSTOM_SYSTEM_PROMPT = '''You are a friendly, conversational assistant who speaks like a colleague over coffee.
Give thorough, step-by-step explanations, including relevant examples or context.
If you make any claims, back them up with evidence. Aim for at least 3–5 sentences per answer.
Format answers so that they’re easy to understand, using paragraphs and headings if needed.
If something isn’t clear, say “I’m not sure, please contact a member of the team.”'''

# Map prompt: applied to each chunk
map_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""{CUSTOM_SYSTEM_PROMPT}

Context:
{{context}}

Question:
{{question}}"""
)

# Combine prompt: merges partial answers
combine_prompt = PromptTemplate(
    input_variables=["summaries", "question"],
    template=f"""{CUSTOM_SYSTEM_PROMPT}

You have the following partial answers from different sections:
{{summaries}}

Combine them into one coherent, detailed answer to the question:
{{question}}"""
)

# ——————————————————————————————
# 💬 Render chat history with styled bubbles
# ——————————————————————————————
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**🤖 Assistant:** {msg['content']}")

# ——————————————————————————————
# ✍️ New user input
# ——————————————————————————————
user_input = st.chat_input("Type your question here…")
if user_input:
    # record & display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    # process and generate response
    with st.spinner("Thinking…"):
        docs = st.session_state.vectorstore.similarity_search(user_input)
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
            openai_api_key=api_key
        )
        chain = load_qa_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt
        )
        answer = chain.run(input_documents=docs, question=user_input)

    # record & display assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(f"**🤖 Assistant:** {answer}")
