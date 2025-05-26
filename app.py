import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ――― Page config & custom CSS ―――
st.set_page_config(page_title="BCCoE CBA Guide Assistant")
st.markdown(
    """
    <style>
      /* shrink headings */
      h1, h2 { font-size: 1.25rem !important; }
      h3      { font-size: 1.1rem  !important; }
      /* style the chat input box */
      .stChatInput > div > div { background-color: #f0f2f6; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("BCCoE CBA Guide Assistant")

# ――― Welcome banner ―――
st.markdown(
    """
    <div style="background-color:#333333; color:white; padding:15px; border-radius:8px; margin-bottom:20px">
      <h4 style="margin:0">👋 Welcome!</h4>
      <p style="margin:5px 0 0">
        Ask me anything about the Cost–Benefit Analysis Guides and I’ll help you step by step.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ――― Initialise chat history ―――
if "messages" not in st.session_state:
    st.session_state.messages = []

# ――― Load API key & vectorstore once ―――
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("🔑 Missing OpenAI API key.")
    st.stop()

if "vectorstore" not in st.session_state:
    raw_text = ""
    for fn in os.listdir("data"):
        if fn.lower().endswith(".pdf"):
            path = os.path.join("data", fn)
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text()
                    if txt:
                        raw_text += txt + "\n"

    # split into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(raw_text)

    # embed & index
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    st.session_state.vectorstore = DocArrayInMemorySearch.from_texts(
        chunks, embedding=embeddings
    )

# ――― System prompt & template ―――
SYSTEM_PROMPT = """You are a friendly, conversational assistant and an expert guide on cost–benefit analysis (CBA).
Help users understand and apply the CBA Guides step by step, drawing only on its methodologies and examples.
If you reference any principle or calculation, cite the relevant section or example from the Guide.
Aim for clear, concise explanations of at least 3–5 sentences, including worked examples where helpful.
Structure your answer in Markdown:
- **# Heading:** to introduce the topic
- **## Subheadings:** for key steps or concepts
- **Bullet lists** or **numbered steps** for procedures
- **Bold** for definitions, _italics_ for emphasis
- Code or formula blocks (triple backticks) for numerical examples

If you can’t answer from the Guide, say: “I’m not sure based on the guides — please check the relevant guide or contact a team member.”"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""{SYSTEM_PROMPT}

Context:
{{context}}

Question:
{{question}}"""
)

# ――― Render chat history ―――
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role):
        # user: bold name, assistant: raw markdown
        if role == "user":
            st.markdown(f"**🧑 You:** {content}")
        else:
            st.markdown(content)

# ――― New user input ―――
user_input = st.chat_input("Type your question here…")
if user_input:
    # record & show
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_input}")

    # generate
    with st.spinner("Thinking…"):
        docs = st.session_state.vectorstore.similarity_search(user_input)
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,  # 0.0 = deterministic, 1.0 = creative
            top_p=0.9,
            max_tokens=700,
            openai_api_key=api_key,
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        answer = chain.run(input_documents=docs, question=user_input)

    # record & display assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
