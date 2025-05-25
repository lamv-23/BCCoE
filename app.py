import os
import streamlit as st
from PyPDF2 import PdfReader
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai.error import RateLimitError

# ——————————————————————————————
# 📄 Page config & custom CSS
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")
st.markdown(
    """
    <style>
      h1, h2 { font-size: 1.25rem !important; }
      h3 { font-size: 1.1rem !important; }
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
# 🛠️ Retry wrapper for embedding step
# ——————————————————————————————
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
)
def embed_documents_retry(embeddings, texts):
    return embeddings.embed_documents(texts)

# ——————————————————————————————
# 📦 Load & cache vectorstore once
# ——————————————————————————————
@st.cache_resource
def load_vectorstore(api_key: str):
    # read all PDFs
    full_text = ""
    for fn in os.listdir("data"):
        if fn.lower().endswith(".pdf"):
            with open(os.path.join("data", fn), "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    full_text += txt + "\n"

    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = splitter.split_text(full_text)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # use our retry wrapper here
    vectors = DocArrayInMemorySearch.from_texts(
        chunks, embedding=embeddings, embedding_function=lambda ts: embed_documents_retry(embeddings, ts)
    )
    return vectors

# ——————————————————————————————
# 🔑 Initialise API key & vectorstore
# ——————————————————————————————
api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

vectorstore = load_vectorstore(api_key)

# ——————————————————————————————
# 🧠 Prompt setup
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

# new input
user_input = st.chat_input("Type your question here…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    with st.spinner("Thinking…"):
        docs = vectorstore.similarity_search(user_input)
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,  # controls randomness/creativity: 0.0 = deterministic, 1.0 = creative
            top_p=0.9,
            max_tokens=700,   # caps length of response
            openai_api_key=api_key
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        answer = chain.run(input_documents=docs, question=user_input)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)
