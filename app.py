import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ——————————————————————————————
# 📄 Streamlit page config & CSS
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")
st.markdown("""
    <style>
      h1, h2 {font-size:1.25rem !important;}
      h3     {font-size:1.1rem  !important;}
      /* style the user input box */
      .stTextInput > div > div > input {
        background-color: #f0f0f5;
        border-radius: 8px;
      }
    </style>
""", unsafe_allow_html=True)

st.title("BCCoE CBA Guide Assistant")

st.markdown("""
<div style="background-color:#333; color:white; padding:15px; border-radius:8px; margin-bottom:20px">
  <h4 style="margin:0">👋 Welcome to the CBA Guide Assistant</h4>
  <p style="margin:5px 0 0">
    Ask me anything about cost–benefit analysis guides and I'll help you out.
  </p>
</div>
""", unsafe_allow_html=True)

# ——————————————————————————————
# 💬 Initialise chat history
# ——————————————————————————————
if "messages" not in st.session_state:
    st.session_state.messages = []

# ——————————————————————————————
# 🔑 Load API key & build FAISS index (once)
# ——————————————————————————————
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("🔑 Please provide your OpenAI API key in Streamlit secrets or as OPENAI_API_KEY.")
    st.stop()

if "vectorstore" not in st.session_state:
    full_text = ""
    for fn in os.listdir("data"):
        if fn.lower().endswith(".pdf"):
            path = os.path.join("data", fn)
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text()
                    if txt:
                        full_text += txt + "\n"

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(full_text)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)

# ——————————————————————————————
# 🧠 Prompt & LLMChain setup
# ——————————————————————————————
SYSTEM_PROMPT = """You are a friendly, conversational assistant and expert in cost–benefit analysis (CBA).  
Help users apply the CBA guides step by step, drawing only on those methodologies and examples.  
If you reference any principle or calculation, cite the relevant section or example.  
Aim for clear explanations of 3–5 sentences, with worked examples where helpful.  
Structure your answer in Markdown:
- **# Heading:** introduce the topic  
- **## Subheadings:** for key steps  
- **Bullet** or **numbered** lists for procedures  
- **Bold** for definitions, _italics_ for emphasis  
- ```formula``` blocks for numerical examples  

If you can’t answer, say: “I’m not sure—please check the guide or contact a team member.”"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{SYSTEM_PROMPT}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}"
)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    temperature=0.2,   # controls randomness; 0.0 = deterministic, 1.0 = very creative
    top_p=0.9,
    max_tokens=700,
    openai_api_key=api_key
)
chain = LLMChain(llm=llm, prompt=prompt)

# ——————————————————————————————
# 💬 Render existing chat
# ——————————————————————————————
for msg in st.session_state.messages:
    avatar = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(avatar):
        # user messages get a prefix
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(msg["content"], unsafe_allow_html=False)

# ——————————————————————————————
# ✍️ Accept user question
# ——————————————————————————————
user_q = st.chat_input("Type your question here…")
if user_q:
    # log & show
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_q}")

    # retrieve & answer
    with st.spinner("Thinking…"):
        docs = st.session_state.vectorstore.similarity_search(user_q, k=5)
        context = "\n\n".join(d.page_content for d in docs)
        answer = chain.run(context=context, question=user_q)

    # log & display assistant
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)
