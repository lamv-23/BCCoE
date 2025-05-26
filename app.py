import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ——————————————————————————————
# 📄 Streamlit page config & CSS
# ——————————————————————————————
st.set_page_config(page_title="BCCoE CBA Guide Assistant")
st.markdown("""
    <style>
      h1, h2 {font-size:1.25rem !important;}
      h3   {font-size:1.1rem  !important;}
      /* style the user input box */
      .stTextInput > div > div > input {
        background-color: #f0f0f5;
        border-radius: 8px;
      }
    </style>
""", unsafe_allow_html=True)

st.title("BCCoE CBA Guide Assistant")

st.markdown("""
<div style="background-color:#333; color:white; padding:15px; border-radius:8px; margin-bottom:20px">
  <h4 style="margin:0">👋 Welcome to the CBA Guide Assistant</h4>
  <p style="margin:5px 0 0">
    Ask me anything about cost–benefit analysis guides and I'll help you out.
  </p>
</div>
""", unsafe_allow_html=True)

# ——————————————————————————————
# 🔄 Clear chat button
# ——————————————————————————————
if st.button("🔄 Clear Chat History"):
    st.session_state.messages = []
    # rerun so the cleared chat is immediately reflected
    st.experimental_rerun()

# ——————————————————————————————
# 💬 Initialise chat history
# ——————————————————————————————
if "messages" not in st.session_state:
    st.session_state.messages = []

# ——————————————————————————————
# 🔑 Load API key & build FAISS index (once)
# ——————————————————————————————
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("🔑 Please provide your OpenAI API key in Streamlit secrets or as OPENAI_API_KEY environment variable.")
    st.stop()

if "vectorstore" not in st.session_state:
    full_text = ""
    data_dir = "data" # Define data directory
    if not os.path.exists(data_dir):
        st.error(f"Data directory '{data_dir}' not found. Please ensure your PDF files are in a folder named 'data'.")
        st.stop()

    pdf_found = False
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".pdf"):
            pdf_found = True
            path = os.path.join(data_dir, fn)
            try:
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        txt = page.extract_text()
                        if txt:
                            full_text += txt + "\n"
            except Exception as e:
                st.warning(f"Could not read PDF file {fn}: {e}")
                continue

    if not pdf_found:
        st.error(f"No PDF files found in '{data_dir}'. Please upload your CBA guide PDFs to this directory.")
        st.stop()
    if not full_text:
        st.error("Could not extract any text from the provided PDF files. Please check the PDFs for text content.")
        st.stop()


    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=,1000
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(full_text)

    # Handle cases where no chunks are generated
    if not chunks:
        st.error("No text chunks could be generated from the PDF content. This might indicate issues with text extraction or very short documents.")
        st.stop()

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    try:
        st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        st.error(f"Failed to create FAISS vectorstore. Please check your OpenAI API key and the content of your PDFs. Error: {e}")
        st.stop()


# ——————————————————————————————
# 🧠 Prompt & LLMChain setup
# ——————————————————————————————
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant and expert teacher on cost–benefit analysis (CBA) guides.
Always base your answers on the information from the provided PDFs, using your best judgement to summarise, explain, or clarify.
If a user asks about numbers, parameters, tables, or formulas, present them clearly (e.g. as Markdown tables or in a list), and reference where in the guides they come from if possible.
Use headings, subheadings, bullet points, and short paragraphs to make answers clear and easy to read.
If the answer is not clearly available in the PDFs, explain that politely, e.g. "I couldn't find this information in the provided guides."
You may add brief, expert explanations to help users understand the material, just as a helpful teacher would.
"""



# The prompt now includes chat_history
prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=f"{SYSTEM_PROMPT}\n\nChat History:\n{{chat_history}}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}"
)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    temperature=0.3,    # comment: controls randomness; 0.0 = deterministic
    top_p=0.9,
    max_tokens=700,
    openai_api_key=api_key
)
chain = LLMChain(llm=llm, prompt=prompt)

# ——————————————————————————————
# 💬 Render existing chat
# ——————————————————————————————
for msg in st.session_state.messages:
    avatar = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(avatar):
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(msg["content"], unsafe_allow_html=False) # Keep unsafe_allow_html=False if you want to prevent arbitrary HTML from LLM

# ——————————————————————————————
# ✍️ Accept user question
# ——————————————————————————————
user_q = st.chat_input("Type your question here…")
if user_q:
    # log & show user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_q}")

    # Prepare chat history for the LLM
    # We only include previous user/assistant messages, excluding the current user question
    formatted_chat_history = ""
    for msg in st.session_state.messages[:-1]: # Exclude the very last message (the current user_q)
        if msg["role"] == "user":
            formatted_chat_history += f"User: {msg['content']}\n"
        else:
            formatted_chat_history += f"Assistant: {msg['content']}\n"

    # retrieve & answer
    with st.spinner("Thinking…"):
        docs = st.session_state.vectorstore.similarity_search(user_q, k=7) # k=5 is a good starting point
        context = "\n\n".join(d.page_content for d in docs)

        try:
            answer = chain.run(context=context, question=user_q, chat_history=formatted_chat_history)
        except Exception as e:
            answer = f"I apologize, but I encountered an error while processing your request: {e}. Please try again or rephrase your question."
            st.error(answer)


    # log & display assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False) # Keep unsafe_allow_html=False for security reasons
