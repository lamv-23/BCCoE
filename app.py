import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ——————————————————————————————
# 📄 Page config & CSS
# ——————————————————————————————
st.set_page_config(page_title="CBA Guide Assistant")

st.markdown("""
<style>
  /* shrink markdown headings */
  h1, h2 { font-size:1.25rem !important; }
  h3    { font-size:1.1rem  !important; }
</style>
""", unsafe_allow_html=True)

st.title("CBA Guide Assistant")

# ——————————————————————————————
# 🔑 Load API key & prebuilt FAISS index
# ——————————————————————————————
api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

# load embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# load the persisted index
persist_dir = "faiss_index"
vectorstore = FAISS.load_local(persist_dir, embeddings)

# ——————————————————————————————
# 🧠 Prompt template
# ——————————————————————————————
CUSTOM_SYSTEM_PROMPT = '''You are a friendly assistant and an expert on cost–benefit analysis (CBA)...'''
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""{CUSTOM_SYSTEM_PROMPT}

Context:
{{context}}

Question:
{{question}}"""
)

# ——————————————————————————————
# 💬 Chat UI
# ——————————————————————————————
if "messages" not in st.session_state:
    st.session_state.messages = []

# render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"], unsafe_allow_html=False)
        else:
            st.markdown(f"**🧑 You:** {msg['content']}")

# new user input
user_input = st.chat_input("Type your question here…")
if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    with st.spinner("Thinking…"):
        docs = vectorstore.similarity_search(user_input)
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,  # controls randomness/creativity
            top_p=0.9,
            max_tokens=700,
            openai_api_key=api_key
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        answer = chain.run(input_documents=docs, question=user_input)

    st.session_state.messages.append({"role":"assistant","content":answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)
