import os
import re
import streamlit as st
import pandas as pd
import pdfplumber
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ——————————————————————————————
# 📄 Page config & custom CSS
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")
st.markdown("""
  <style>
    h1, h2 { font-size:1.25rem!important; }
    h3     { font-size:1.1rem!important; }
    div[data-testid="stChatInputContainer"] textarea {
      background-color:#e0f7fa!important;
      border-radius:4px!important;
    }
  </style>
""", unsafe_allow_html=True)

st.title("BCCoE CBA Guide Assistant")
st.markdown("""
  <div style="background-color:#333; color:white; padding:15px; border-radius:8px; margin-bottom:20px">
    <h4 style="margin:0">👋 Welcome to the CBA Guide Assistant</h4>
    <p style="margin:5px 0 0">Ask me anything about cost–benefit analysis guides and I'll do my best to help.</p>
  </div>
""", unsafe_allow_html=True)

# initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ——————————————————————————————
# 🔑 Load & cache vector store + tables
# ——————————————————————————————
@st.cache_resource
def init_store_and_tables(api_key: str):
    file_texts = {}
    pdf_tables = {}

    for fn in sorted(os.listdir("data")):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join("data", fn)
        # Extract text
        reader = PdfReader(path)
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        file_texts[fn] = text

        # Extract tables
        tables = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for raw_tbl in page.extract_tables():
                    if len(raw_tbl) < 2:
                        continue
                    header = raw_tbl[0]
                    # Clean duplicates
                    counts = {}
                    cols = []
                    for h in header:
                        name = (h or "column").strip()
                        c = counts.get(name, 0)
                        counts[name] = c + 1
                        key = f"{name}_{c}" if c else name
                        cols.append(re.sub(r"\W+", "_", key).lower())
                    df = pd.DataFrame(raw_tbl[1:], columns=cols)
                    tables.append(df)
        pdf_tables[fn] = tables

    # Split + metadata
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=3500, chunk_overlap=100, length_function=len
    )
    chunks, metadatas = [], []
    for fn, text in file_texts.items():
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            metadatas.append({"source": fn})

    # Build vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vs = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings, metadatas=metadatas)
    return vs, pdf_tables

api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

vectorstore, pdf_tables = init_store_and_tables(api_key)

# ——————————————————————————————
# 🧠 PromptTemplate & system prompt
# ——————————————————————————————
CUSTOM_SYSTEM_PROMPT = '''You are a friendly, expert assistant on Cost–Benefit Analysis Guides.
Use only the provided excerpts—each chunk is tagged with its source guide.
Answer in 3–5 sentences minimum, include worked examples, and format in Markdown:
- `# Heading` for topics
- `## Subheading` for steps
- Bullet/numbered lists
- **Bold** definitions, _italics_ for emphasis
- ```code blocks``` for formulas.

If you can’t answer, say:
“I’m not sure based on the guides—please check the relevant section or contact a team member.”'''

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""{CUSTOM_SYSTEM_PROMPT}

Context:
{{context}}

Question:
{{question}}"""
)

# ——————————————————————————————
# Render chat history
# ——————————————————————————————
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"], unsafe_allow_html=False)
        else:
            st.markdown(f"**🧑 You:** {msg['content']}")

# ——————————————————————————————
# Handle new input
# ——————————————————————————————
user_input = st.chat_input("Type your question here…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    with st.spinner("Thinking…"):
        # retrieve top-5 docs
        docs = vectorstore.similarity_search(user_input, k=5)

        # build one big context string
        big_context = "\n\n---\n\n".join(
            f"**From {d.metadata['source']}:**\n{d.page_content}"
            for d in docs
        )

        # call LLMChain
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,
            top_p=0.9,
            max_tokens=700,
            openai_api_key=api_key
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        answer = chain.run({"context": big_context, "question": user_input})

    # append & display
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)

    # only show tables when question mentions “table”
    if "table" in user_input.lower():
        st.subheader("Related Tables from Guides")
        shown = set()
        for d in docs:
            src = d.metadata["source"]
            if src in shown:
                continue
            shown.add(src)
            tbls = pdf_tables.get(src, [])
            if not tbls:
                continue
            with st.expander(f"Tables in {src}"):
                for i, df in enumerate(tbls, 1):
                    st.write(f"**Table {i}:**")
                    st.dataframe(df, use_container_width=True)
