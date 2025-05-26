import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# ——————————————————————————————
# 📄 Page config & CSS
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")
st.markdown(
    """
    <style>
      h1, h2 { font-size: 1.25rem !important; }
      h3      { font-size: 1.1rem  !important; }
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
    st.session_state.messages = []

# ——————————————————————————————
# 🔑 Load API key & build vector store
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
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    st.session_state.vectorstore = DocArrayInMemorySearch.from_texts(
        chunks, embedding=embeddings
    )

# ——————————————————————————————
# 🧠 SYSTEM prompt (merged full instructions + numeric‐table rule)
# ——————————————————————————————
SYSTEM = """
You are a friendly, conversational assistant who speaks like a colleague over coffee and an expert guide on cost–benefit analysis (CBA).
Help users understand and apply the CBA Guides step by step, drawing only on its methodologies and examples.
If you reference any principle or calculation, cite the relevant section or example from the Guide.
Aim for clear, concise explanations of at least 3–5 sentences per answer.
Structure your answer in Markdown:
- **# Heading:** to introduce the topic
- **## Subheadings:** for key steps or concepts
- **Bullet lists** or **numbered steps** for procedures
- **Bold** for definitions, _italics_ for emphasis
- Code or formula blocks (triple backticks) for numerical examples

**Special rule for numeric‐parameter questions**:
Whenever the user asks for **numeric parameter values** (e.g. “value of time for NSW projects”):
1. Look up the exact dollar figures or ranges from the guides.
2. Present them as a **Markdown table** with these headers:

   | Parameter     | Value  | Units  | Source                       |
   | ------------- | ------ | ------ | ---------------------------- |
   | Value of Time | $27.00 | $/hr   | NSW CBA Guide, Section 4.2   |

If you can’t find the numbers, respond:

> I’m not sure based on the guides—please check the relevant guide or contact a team member.
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{SYSTEM}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}",
)

# ——————————————————————————————
# 💬 Render chat history
# ——————————————————————————————
for msg in st.session_state.messages:
    avatar = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(avatar):
        if msg["role"] == "assistant":
            st.markdown(msg["content"], unsafe_allow_html=False)
        else:
            st.markdown(f"**🧑 You:** {msg['content']}")

# ——————————————————————————————
# ✍️ New user input
# ——————————————————————————————
user_input = st.chat_input("Type your question here…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    with st.spinner("Thinking…"):
        docs = st.session_state.vectorstore.similarity_search(user_input, k=3)
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,    # controls randomness; 0.0 = deterministic
            top_p=0.9,
            max_tokens=700,
            openai_api_key=api_key,
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
        answer = chain.run(input_documents=docs, question=user_input)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)
