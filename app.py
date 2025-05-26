import os
import re
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
      <h4 style="margin:0">👋 Welcome to the CBA Guide Assistant</h4>
      <p style="margin:5px 0 0">
        Ask me anything about cost–benefit analysis guides and I'll do my best to help.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ——————————————————————————————
# 🔑 Load API key & vector store once, plus table extraction
# ——————————————————————————————
@st.cache_resource
def init_store_and_tables(api_key: str):
    # 1) Read & combine text, extract tables
    all_text = ""
    pdf_tables: dict[str, list[pd.DataFrame]] = {}
    file_texts: dict[str, str] = {}

    for fn in sorted(os.listdir("data")):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join("data", fn)

        # — Extract full text
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        file_texts[fn] = text
        all_text += text + "\n"

        # — Extract tables with pdfplumber, clean headers
        tables: list[pd.DataFrame] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for raw_tbl in page.extract_tables():
                    if not raw_tbl or len(raw_tbl) < 2:
                        continue
                    raw_header = raw_tbl[0]
                    header_counts: dict[str,int] = {}
                    clean_header: list[str] = []
                    for col in raw_header:
                        # default name for missing headers
                        name = col.strip() if isinstance(col, str) and col.strip() else "column"
                        # de-duplicate
                        count = header_counts.get(name, 0)
                        header_counts[name] = count + 1
                        if count:
                            name = f"{name}_{count}"
                        # snake_case & lowercase
                        name = re.sub(r"\W+", "_", name).lower()
                        clean_header.append(name)
                    df = pd.DataFrame(raw_tbl[1:], columns=clean_header)
                    tables.append(df)
        pdf_tables[fn] = tables

    # 2) Split each file's text into chunks, tag with metadata
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3500,
        chunk_overlap=100,
        length_function=len
    )
    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    for fn, text in file_texts.items():
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": fn})

    # 3) Embed and build vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = DocArrayInMemorySearch.from_texts(
        all_chunks, embedding=embeddings, metadatas=all_metadatas
    )

    return vectorstore, pdf_tables

api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

vectorstore, pdf_tables = init_store_and_tables(api_key)

# ——————————————————————————————
# 📋 Show extracted tables
# ——————————————————————————————
if any(pdf_tables.values()):
    st.subheader("Extracted Tables")
    for fn, tables in pdf_tables.items():
        if not tables:
            continue
        with st.expander(f"Tables in {fn}"):
            for i, df in enumerate(tables, start=1):
                st.write(f"**Table {i}:**")
                st.dataframe(df, use_container_width=True)

# ——————————————————————————————
# 🧠 System prompt & PromptTemplate
# ——————————————————————————————
CUSTOM_SYSTEM_PROMPT = '''You are a friendly, conversational assistant and expert on Cost–Benefit Analysis (CBA) Guides.  
Use only the provided excerpts—each chunk is tagged with its source guide.  
When you answer, mention the guide by name, for example:
> “According to *{source}* (section …), …”

Aim for 3–5 sentences minimum, include worked examples, and format in Markdown:
- `# Heading` to introduce topics  
- `## Subheading` for steps or concepts  
- Bullet or numbered lists for procedures  
- **Bold** for definitions, _italics_ for emphasis  
- ```code blocks``` for formulas or numerical examples  

If the guide doesn’t cover the question, say:
> “I’m not sure based on the guides — please check the relevant section or contact a team member.”'''

prompt = PromptTemplate(
    input_variables=["context", "question", "source"],
    template=f"""{CUSTOM_SYSTEM_PROMPT}

Context (from {{source}}):
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
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    with st.spinner("Thinking…"):
        # retrieve top‐5 chunks
        docs = vectorstore.similarity_search(user_input, k=5)

        # build a combined context string that injects the source name
        contexts, sources = [], []
        for d in docs:
            contexts.append(d.page_content)
            sources.append(d.metadata.get("source", "unknown"))
        # for simplicity, send only the first source in prompt; chain will have all contexts
        chain_input = {
            "context": "\n\n".join(contexts),
            "question": user_input,
            "source": sources[0]
        }

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,
            top_p=0.9,
            max_tokens=700,
            openai_api_key=api_key
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        answer = chain.run(**chain_input)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)
