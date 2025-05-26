import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# ——————————————————————————————
# 📄 Page config & CSS to shrink headings
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")
st.markdown(
    """
    <style>
      h1, h2 { font-size: 1.25rem !important; }
      h3       { font-size: 1.1rem  !important; }
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
    st.session_state.messages = []  # each message is {"role": "user"/"assistant", "content": text}

# ——————————————————————————————
# 🔑 Load API key & build vector store (once)
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

    # 2) Chunk
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # 3) Embed & store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    st.session_state.vectorstore = DocArrayInMemorySearch.from_texts(
        chunks, embedding=embeddings
    )

# ——————————————————————————————
# 🧠 Define prompts
# ——————————————————————————————
CUSTOM_SYSTEM_PROMPT = """You are a friendly, conversational assistant and an expert guide on cost–benefit analysis (CBA).
Help users understand and apply the CBA Guides step by step, drawing only on its methodologies and examples.
If you reference any principle or calculation, cite the relevant section or example from the Guide.
Aim for clear, concise explanations of at least 3–5 sentences, including worked examples where helpful.
Structure your answer in Markdown:
- **# Heading:** to introduce the topic
- **## Subheadings:** for key steps or concepts
- Bullet lists or numbered steps for procedures
- **Bold** for definitions, _italics_ for emphasis
- Code or formula blocks (triple backticks) for numerical examples

If you can’t answer from the Guide, say: “I’m not sure based on the guides—please check the relevant guide or contact a team member.”"""

# Used by the “stuff” chain inside the ConversationalRetrievalChain
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{CUSTOM_SYSTEM_PROMPT}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}"
)

# Optional: rephrase follow-ups into standalone questions
condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "Given the following conversation:\n{chat_history}\n"
        "Rephrase the last user question so it's standalone:\n\"{question}\""
    )
)

# ——————————————————————————————
# 💬 Render chat history
# ——————————————————————————————
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    avatar = "user" if role == "user" else "assistant"
    with st.chat_message(avatar):
        # render assistant in full Markdown
        if role == "assistant":
            st.markdown(content, unsafe_allow_html=False)
        else:
            st.markdown(f"**🧑 You:** {content}")

# ——————————————————————————————
# ✍️ New user input
# ——————————————————————————————
user_input = st.chat_input("Type your question here…")
if user_input:
    # 1) Log & display user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    # 2) Build & run conversational chain
    with st.spinner("Thinking…"):
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,    # comment: controls randomness; 0.0 = fully deterministic
            top_p=0.9,
            max_tokens=700,
            openai_api_key=api_key
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            condense_question_prompt=condense_prompt,
            qa_prompt=qa_prompt,
        )
        result = qa_chain({
            "question": user_input,
            "chat_history": st.session_state.messages
        })
        answer = result["answer"]

    # 3) Log & display assistant
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)
