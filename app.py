import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# ——————————————————————————————
# 📄 Page config & CSS tweaks
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")
st.markdown(
    """
    <style>
      /* shrink H1/H2/H3 */
      h1, h2 { font-size: 1.25rem !important; }
      h3    { font-size: 1.1rem  !important; }

      /* style chat input box */
      .stChatInput textarea {
        background-color: #f0f0f0 !important;
        border-radius: 8px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("BCCoE CBA Guide Assistant")

# ——————————————————————————————
# 👋 Welcome banner
# ——————————————————————————————
st.markdown(
    """
    <div style="background-color:#333333; color:white; padding:15px; border-radius:8px; margin-bottom:20px">
      <h4 style="margin:0">👋 Welcome to the BCCoE CBA Guide Assistant</h4>
      <p style="margin:5px 0 0">
        Ask me anything about cost–benefit analysis guides and I'll help you out.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ——————————————————————————————
# 💬 Initialise chat history
# ——————————————————————————————
if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role": "user"/"assistant", "content": text}

# ——————————————————————————————
# 🔑 Load API key & build FAISS vector store (once)
# ——————————————————————————————
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

if "vectorstore" not in st.session_state:
    # 1) Read all PDFs in data/
    text = ""
    for fn in os.listdir("data"):
        if fn.lower().endswith(".pdf"):
            with open(os.path.join("data", fn), "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        text += txt + "\n"

    # 2) Chunk into ~1k-character pieces
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # 3) Embed & FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)

# ——————————————————————————————
# 🧠 Define your system + QA prompts
# ——————————————————————————————
SYSTEM = """You are a friendly, conversational assistant and an expert guide on cost–benefit analysis (CBA).
Help users understand and apply the CBA Guides step by step, drawing only on its methodologies and examples.
If you reference any principle or calculation, cite the relevant section or example from the Guide.
Aim for clear, concise explanations of at least 3–5 sentences, including worked examples where helpful.
Structure your answer in Markdown:
- **# Heading:** to introduce the topic
- **## Subheadings:** for key steps or concepts
- Bullet lists or numbered steps for procedures
- **Bold** for definitions, _italics_ for emphasis
- Code or formula blocks (triple backticks) for numerical examples

If you can’t answer from the Guide, say:
“I’m not sure based on the guides—please check the relevant guide or contact a team member.”"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{SYSTEM}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}"
)

condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "Given the following conversation:\n{chat_history}\n"
        "Rephrase the last user question so it's standalone:\n\"{question}\""
    )
)

# ——————————————————————————————
# 💬 Render existing chat messages
# ——————————————————————————————
for msg in st.session_state.messages:
    avatar = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(avatar):
        if msg["role"] == "assistant":
            st.markdown(msg["content"], unsafe_allow_html=False)
        else:
            st.markdown(f"**🧑 You:** {msg['content']}")

# ——————————————————————————————
# ✍️ Handle new user input
# ——————————————————————————————
user_input = st.chat_input("Type your question here…")
if user_input:
    # 1) Log & display the user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    # 2) Build & run the conversational chain
    with st.spinner("Thinking…"):
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,    # controls randomness: 0.0 = deterministic
            top_p=0.9,
            max_tokens=700,
            openai_api_key=api_key
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            condense_question_prompt=condense_prompt,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt},
        )
        result = qa_chain({
            "question": user_input,
            "chat_history": st.session_state.messages
        })
        answer = result["answer"]

    # 3) Log & display the assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)
