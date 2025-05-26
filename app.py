import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ——————————————————————————————
# 📄 Streamlit page config & CSS
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")
st.markdown(
    """
    <style>
      /* Shrink markdown headings */
      h1, h2 { font-size: 1.25rem !important; }
      h3      { font-size: 1.1rem  !important; }
      /* Colour the input box */
      .st-TextInput>div>div>input {
        background-color: #f0f0f5 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("BCCoE CBA Guide Assistant")

# Welcome banner
st.markdown(
    """
    <div style="background-color:#333333; color:white; padding:15px; border-radius:8px; margin-bottom:20px">
      <h4 style="margin:0">👋 Welcome to the CBA Guide Assistant</h4>
      <p style="margin:5px 0 0">
        Ask me anything about cost–benefit analysis guides.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ——————————————————————————————
# 🔑 Load API key & build vector store once
# ——————————————————————————————
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

if "vectorstore" not in st.session_state:
    # Read all PDFs under data/
    combined_text = ""
    for fn in os.listdir("data"):
        if fn.lower().endswith(".pdf"):
            with open(os.path.join("data", fn), "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    combined_text += (page.extract_text() or "") + "\n"

    # Split into chunks
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = splitter.split_text(combined_text)

    # Create embeddings & vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    st.session_state.vectorstore = DocArrayInMemorySearch.from_texts(
        chunks, embedding=embeddings
    )

# ——————————————————————————————
# 🧠 Custom system prompt & PromptTemplate
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
# 💬 Initialise chat history
# ——————————————————————————————
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # assistant content is already markdown
        if msg["role"] == "assistant":
            st.markdown(msg["content"], unsafe_allow_html=False)
        else:
            st.markdown(f"**🧑 You:** {msg['content']}")

# ——————————————————————————————
# ✍️ Handle new user input
# ——————————————————————————————
user_input = st.chat_input("Type your question here…")
if user_input:
    # Echo user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    # Retrieval + LLMChain
    with st.spinner("Thinking…"):
        # 1) find relevant chunks
        docs = st.session_state.vectorstore.similarity_search(user_input)
        # 2) combine into one context string
        context = "\n\n".join(d.page_content for d in docs)
        # 3) instantiate LLM and chain
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,   # lower = more focused, higher = more creative
            top_p=0.9,
            max_tokens=700,
            openai_api_key=api_key
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        # 4) generate answer
        answer = chain.predict(context=context, question=user_input)

    # Display assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)
