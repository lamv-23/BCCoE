import os
import streamlit as st
import pandas as pd
import pdfplumber
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ——————————————————————————————
# 📄 Page config & custom CSS
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")

st.markdown(
    """
    <style>
      h1, h2 { font-size: 1.25rem !important; }
      h3     { font-size: 1.1rem  !important; }
      div[data-testid="stChatInputContainer"] textarea {
        background-color: #e0f7fa !important;
        color: #000 !important;
        border-radius: 4px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("BCCoE CBA Guide Assistant")

st.markdown(
    """
    <div style="background-color:#333333; color:white; padding:15px; border-radius:8px; margin-bottom:20px">
      <h4 style="margin:0">👋 Welcome to the BCCoE CBA Guide Assistant</h4>
      <p style="margin:5px 0 0">
        Ask me anything about developing CBAs and I'll do my best to help.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ——————————————————————————————
# 🔑 Load API key
# ——————————————————————————————
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

# ——————————————————————————————
# 🚀 Build & cache the vectorstore and extract tables once
# ——————————————————————————————
@st.cache_resource
def init_store_and_tables(api_key: str):
    # 1) Read & combine all PDF text
    all_text = ""
    pdf_tables: dict[str, list[pd.DataFrame]] = {}

    for fn in sorted(os.listdir("data")):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join("data", fn)

        # — Extract text for embedding
        reader = PdfReader(path)
        for page in reader.pages:
            txt = page.extract_text() or ""
            all_text += txt + "\n"

        # — Extract tables via pdfplumber
        tables = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for tbl in page.extract_tables():
                    # first row as header
                    if not tbl or len(tbl) < 2:
                        continue
                    df = pd.DataFrame(tbl[1:], columns=tbl[0])
                    tables.append(df)
        pdf_tables[fn] = tables

    # 2) Split text into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_text(all_text)

    # 3) Embed & build vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings)

    return vectorstore, pdf_tables

vectorstore, pdf_tables = init_store_and_tables(api_key)

# ——————————————————————————————
# 📊 Display any extracted tables
# ——————————————————————————————
if any(pdf_tables.values()):
    st.subheader("📋 Extracted Tables")
    for fn, tables in pdf_tables.items():
        if not tables:
            continue
        with st.expander(f"Tables found in {fn}"):
            for i, df in enumerate(tables, start=1):
                st.write(f"**Table {i}:**")
                st.dataframe(df, use_container_width=True)

# ——————————————————————————————
# 🧠 Define custom prompt for detail & Markdown
# ——————————————————————————————
CUSTOM_SYSTEM_PROMPT = '''You are a friendly, conversational assistant and an expert guide on cost–benefit analysis (CBA).  
Help users understand and apply the CBA Guides step by step, drawing only on its methodologies and examples.  
If you reference any principle or calculation, cite the relevant section or example from the Guide.  
Aim for clear, concise explanations of at least 3–5 sentences, including worked examples where helpful.  
Structure your answer in Markdown:
- **# Heading:** to introduce the topic  
- **## Subheadings:** for key steps or concepts  
- **Bullet lists** or **numbered steps** for procedures  
- **Bold** for definitions, _italics_ for emphasis  
- Code or formula blocks (triple backticks) for numerical examples  

If you can’t answer from the Guide, say: “I’m not sure based on the guides — please check the relevant guide or contact a team member.”'''

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
        if msg["role"] == "assistant":
            st.markdown(msg["content"], unsafe_allow_html=False)
        else:
            st.markdown(f"**🧑 You:** {msg['content']}")

# ——————————————————————————————
# ✍️ Handle new user input
# ——————————————————————————————
user_input = st.chat_input("Type your question here…")
if user_input:
    # record & display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    # process and generate response
    with st.spinner("Thinking…"):
        docs = vectorstore.similarity_search(user_input)
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,  # controls randomness/creativity: 0.0 = deterministic, 1.0 = very creative
            top_p=0.9,
            max_tokens=700,
            openai_api_key=api_key
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        answer = chain.run(input_documents=docs, question=user_input)

    # record & display assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)
