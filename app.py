import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ——————————————————————————————
# 📄 Page config & custom CSS
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")

st.markdown(
    """
    <style>
      /* Shrink Markdown headings */
      h1, h2 { font-size: 1.25rem !important; }
      h3     { font-size: 1.1rem  !important; }
      /* Style the chat input box */
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

# ——————————————————————————————
# 🔑 Load & cache vectorstore once
# ——————————————————————————————
VECTORSTORE_PATH = "vectorstore.pkl"

@st.cache_resource
def load_vectorstore(api_key: str):
    # If we've already embedded and saved, just load it
    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, "rb") as f:
            return pickle.load(f)

    # Otherwise, read PDFs and split into chunks
    full_text = ""
    for fn in os.listdir("data"):
        if fn.lower().endswith(".pdf"):
            with open(os.path.join("data", fn), "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    full_text += txt + "\n"

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(full_text)

    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings)

    # Persist to disk so we don't re-embed next time
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

# Retrieve API key
api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

# Load (or build) the vectorstore
vectorstore = load_vectorstore(api_key)

# ——————————————————————————————
# 🧠 Define custom prompt for detail & Markdown
# ——————————————————————————————
CUSTOM_SYSTEM_PROMPT = '''You are a friendly, conversational assistant and an expert guide on cost–benefit analysis (CBA).
Help users understand and apply the CBA Guides step by step, drawing only on its methodologies and examples.
If you reference any principle or calculation, cite the relevant section or example from the Guide.
Aim for clear, concise explanations of at least 3–5 sentences, including worked examples where helpful.
Structure your answer in Markdown:
- **# Heading** to introduce the topic
- **## Subheading** for key steps or concepts
- Bullet lists or numbered steps for procedures
- **Bold** for definitions, _italics_ for emphasis
- Code or formula blocks (triple backticks) for numerical examples

If you can’t answer from the Guide, say: “I’m not sure based on the Guide—please check the relevant chapter or contact a team member.”'''

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
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    # log user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    # generate answer
    with st.spinner("Thinking…"):
        docs = vectorstore.similarity_search(user_input)
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,  # controls randomness/creativity: 0.0 = deterministic, 1.0 = very creative
            top_p=0.9,
            max_tokens=700,
            openai_api_key=api_key
        )
        chain = load_qa_chain(
            llm,
            chain_type="stuff",
            prompt=prompt
        )
        answer = chain.run(input_documents=docs, question=user_input)

    # log and display assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)
