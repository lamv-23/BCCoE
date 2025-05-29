import os
import streamlit as st
import pdfplumber
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
    st.rerun()

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

def clean_text(text):
    """Clean and normalize text from PDFs"""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newline
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    return text

def check_pdf_extractable(pdf_path):
    """Check if PDF contains extractable text (not just scanned images)"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Check first few pages for text content
            text_found = False
            for i, page in enumerate(pdf.pages[:3]):  # Check first 3 pages
                text = page.extract_text()
                if text and len(text.strip()) > 50:  # Reasonable amount of text
                    text_found = True
                    break
            return text_found
    except:
        return False

@st.cache_resource
def build_vectorstore():
    """Build and cache the vectorstore to avoid rebuilding on every run"""
    full_text = ""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        st.error(f"Data directory '{data_dir}' not found. Please ensure your PDF files are in a folder named 'data'.")
        st.stop()

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        st.error(f"No PDF files found in '{data_dir}'. Please upload your CBA guide PDFs to this directory.")
        st.stop()
    
    # Check if PDFs are text-extractable
    extractable_pdfs = []
    for fn in pdf_files:
        path = os.path.join(data_dir, fn)
        if check_pdf_extractable(path):
            extractable_pdfs.append(fn)
        else:
            st.warning(f"⚠️ PDF '{fn}' appears to be scanned/image-only. Text extraction may be limited.")

    if not extractable_pdfs:
        st.error("❌ No extractable text found in any PDF files. Please ensure your PDFs contain searchable text, not just scanned images.")
        st.stop()

    for fn in extractable_pdfs:
        path = os.path.join(data_dir, fn)
        try:
            with pdfplumber.open(path) as pdf:
                doc_text = ""
                for page_num, page in enumerate(pdf.pages):
                    txt = page.extract_text()
                    if txt:
                        # Clean up the text
                        cleaned_txt = clean_text(txt)
                        if cleaned_txt:
                            doc_text += f"[Source: {fn}, Page {page_num + 1}]\n{cleaned_txt}\n\n"
                
                if doc_text:
                    full_text += f"\n=== DOCUMENT: {fn} ===\n{doc_text}\n"
                    
        except Exception as e:
            st.warning(f"Could not read PDF file {fn}: {e}")
            continue

    if not full_text:
        st.error("Could not extract any text from the provided PDF files. Please check the PDFs for text content.")
        st.stop()

    # Enhanced text splitting with better parameters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Larger chunks for more context
        chunk_overlap=400,  # More overlap to preserve context
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]  # Better splitting on natural boundaries
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        st.error("No text chunks could be generated from the PDF content.")
        st.stop()

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    try:
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore, len(chunks)
    except Exception as e:
        st.error(f"Failed to create FAISS vectorstore. Error: {e}")
        st.stop()

# Build vectorstore (cached)
vectorstore, total_chunks = build_vectorstore()

# ——————————————————————————————
# 🧠 Improved Prompt & LLMChain setup
# ——————————————————————————————
SYSTEM_PROMPT = """
You are an expert Cost-Benefit Analysis (CBA) consultant and teacher with deep knowledge of government CBA guidelines and best practices.

INTERACTION GUIDELINES:
- For casual greetings (like "Hi", "Hello"), respond warmly and briefly introduce what you can help with
- For general conversation, be friendly and natural
- For CBA-specific questions, provide detailed, expert guidance based on the provided documents

CBA ANALYSIS INSTRUCTIONS (when relevant):
1. **Always prioritize the provided PDF context** - Base your answers primarily on the information from the CBA guide documents
2. **Be specific and detailed** - When referencing numbers, parameters, tables, or formulas, present them clearly with proper formatting
3. **Cite your sources** - When possible, reference which document or section your information comes from
4. **Structure your responses** - Use headings, bullet points, and clear formatting to make answers easy to read
5. **Handle uncertainty honestly** - If information isn't clearly available in the guides, say so explicitly
6. **Provide practical guidance** - Add brief explanations to help users understand and apply the material
7. **Consider context** - Use the chat history to provide consistent, connected responses

RESPONSE FORMAT FOR CBA QUESTIONS:
- Use clear headings and subheadings
- Present numerical data in tables or organized lists
- Include specific page references when available
- Provide actionable guidance where appropriate

If a CBA answer cannot be found in the provided documents, respond with: "I couldn't find this specific information in the provided CBA guides. However, [provide any relevant general guidance if appropriate]."
"""

# Enhanced prompt with better structure
prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=f"""{SYSTEM_PROMPT}

PREVIOUS CONVERSATION:
{{chat_history}}

RELEVANT CONTEXT FROM CBA GUIDES:
{{context}}

CURRENT QUESTION:
{{question}}

RESPONSE:"""
)

def expand_query(query):
    """Expand query with related CBA terms to improve retrieval"""
    cba_synonyms = {
        'cost': ['expense', 'expenditure', 'price', 'financial'],
        'benefit': ['advantage', 'gain', 'value', 'return'],
        'analysis': ['assessment', 'evaluation', 'study', 'review'],
        'discount': ['present value', 'NPV', 'discounting'],
        'risk': ['uncertainty', 'sensitivity', 'probability'],
        'social': ['societal', 'community', 'public'],
        'economic': ['financial', 'monetary', 'fiscal']
    }
    
    expanded_terms = []
    query_words = query.lower().split()
    
    for word in query_words:
        if word in cba_synonyms:
            expanded_terms.extend(cba_synonyms[word][:2])  # Add top 2 synonyms
    
    if expanded_terms:
        return f"{query} {' '.join(expanded_terms[:3])}"  # Add max 3 related terms
    return query

def is_casual_greeting(text):
    """Check if the message is a casual greeting or small talk"""
    casual_patterns = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'what\'s up', 'sup', 'greetings', 'yo'
    ]
    text_lower = text.lower().strip()
    
    # Check if it's a short greeting (under 20 characters and matches patterns)
    if len(text_lower) < 20:
        for pattern in casual_patterns:
            if pattern in text_lower:
                return True
    return False

def handle_casual_greeting():
    """Generate a friendly greeting response"""
    return """Hi there! 👋 

I'm your CBA Guide Assistant, here to help you with cost-benefit analysis questions. I have access to comprehensive CBA guidelines and can help you with:

• Understanding CBA methodology and principles
• Calculating NPV, discount rates, and other metrics  
• Identifying and valuing costs and benefits
• Conducting sensitivity and risk analysis
• Following best practices for government CBAs

Feel free to ask me any specific questions about cost-benefit analysis, or try one of the sample questions in the sidebar to get started!"""
    """Enhanced retrieval with multiple search strategies"""
    all_docs = []
    
    # Strategy 1: Direct similarity search
    primary_docs = vectorstore.similarity_search(query, k=12)
    all_docs.extend(primary_docs)
    
    # Strategy 2: Expanded query search if we need more results
    if len(primary_docs) < 8:
        expanded_query = expand_query(query)
        if expanded_query != query:
            expanded_docs = vectorstore.similarity_search(expanded_query, k=6)
            all_docs.extend(expanded_docs)
    
    # Strategy 3: Keyword-based search for important terms
    important_terms = []
    query_words = query.lower().split()
    for word in query_words:
        if len(word) > 4 and word not in ['what', 'how', 'when', 'where', 'should', 'would', 'could']:
            important_terms.append(word)
    
    # Search for key terms if we still need more context
    if len(all_docs) < 10 and important_terms:
        for term in important_terms[:2]:  # Limit to top 2 terms
            term_docs = vectorstore.similarity_search(term, k=4)
            all_docs.extend(term_docs)
    
    # Remove duplicates while preserving order and relevance
    seen = set()
    unique_docs = []
    for doc in all_docs:
        doc_hash = hash(doc.page_content)
        if doc_hash not in seen and len(unique_docs) < max_chunks:
            seen.add(doc_hash)
            unique_docs.append(doc)
    
def get_relevant_context(vectorstore, query, max_chunks=20):
    """Enhanced retrieval with multiple search strategies"""
    all_docs = []
    
    # Strategy 1: Direct similarity search
    primary_docs = vectorstore.similarity_search(query, k=12)
    all_docs.extend(primary_docs)
    
    # Strategy 2: Expanded query search if we need more results
    if len(primary_docs) < 8:
        expanded_query = expand_query(query)
        if expanded_query != query:
            expanded_docs = vectorstore.similarity_search(expanded_query, k=6)
            all_docs.extend(expanded_docs)
    
    # Strategy 3: Keyword-based search for important terms
    important_terms = []
    query_words = query.lower().split()
    for word in query_words:
        if len(word) > 4 and word not in ['what', 'how', 'when', 'where', 'should', 'would', 'could']:
            important_terms.append(word)
    
    # Search for key terms if we still need more context
    if len(all_docs) < 10 and important_terms:
        for term in important_terms[:2]:  # Limit to top 2 terms
            term_docs = vectorstore.similarity_search(term, k=4)
            all_docs.extend(term_docs)
    
    # Remove duplicates while preserving order and relevance
    seen = set()
    unique_docs = []
    for doc in all_docs:
        doc_hash = hash(doc.page_content)
        if doc_hash not in seen and len(unique_docs) < max_chunks:
            seen.add(doc_hash)
            unique_docs.append(doc)
    
    return unique_docs[:max_chunks]
# Model selection with fallback options
model_options = {
    "GPT-4o Mini (Recommended)": "gpt-4o-mini",  # Cost-effective and good performance
    "GPT-4": "gpt-4",  # Best performance, higher cost
    "GPT-3.5 Turbo": "gpt-3.5-turbo-16k"  # Budget option
}

# Allow user to choose model in sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox(
        "Choose AI Model:",
        options=list(model_options.keys()),
        index=0,  # Default to GPT-4o Mini
        help="GPT-4o Mini offers the best balance of cost and performance for most queries."
    )
    
    max_chunks = st.slider(
        "Context chunks to retrieve:",
        min_value=10,
        max_value=25,
        value=18,
        help="More chunks = more context but higher cost"
    )
    
    st.info(f"📊 Vector store contains {total_chunks} text chunks")

llm = ChatOpenAI(
    model_name=model_options[selected_model],
    temperature=0.1,     # Lower temperature for more consistent responses
    max_tokens=2000,     # Longer responses for detailed answers
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
            st.markdown(msg["content"], unsafe_allow_html=False)

# ——————————————————————————————
# ✍️ Accept user question with improved retrieval
# ——————————————————————————————
user_q = st.chat_input("Type your question here…")
if user_q:
    # Log & show user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(f"**🧑 You:** {user_q}")

    # Prepare chat history (last 6 messages for context without overwhelming the model)
    formatted_chat_history = ""
    recent_messages = st.session_state.messages[-7:-1]  # Last 6 messages, excluding current
    for msg in recent_messages:
        if msg["role"] == "user":
            formatted_chat_history += f"User: {msg['content']}\n"
        else:
            formatted_chat_history += f"Assistant: {msg['content'][:200]}...\n"  # Truncate long responses

    # Check if it's a casual greeting first
    if is_casual_greeting(user_q):
        answer = handle_casual_greeting()
    else:
        # Enhanced retrieval with multiple search strategies
        with st.spinner("🔍 Searching through CBA guides and analyzing context…"):
            try:
                # Get relevant documents using enhanced retrieval
                docs = get_relevant_context(vectorstore, user_q, max_chunks=max_chunks)
                
                if not docs:
                    st.warning("⚠️ No relevant content found. Try rephrasing your question or using different keywords.")
                    answer = "I couldn't find relevant information in the CBA guides for your question. Could you try rephrasing it or asking about a more specific aspect of cost-benefit analysis?"
                else:
                    # Prepare context with clear separation
                    context_parts = []
                    for i, doc in enumerate(docs, 1):
                        context_parts.append(f"--- Context Chunk {i} ---\n{doc.page_content}")
                    
                    context = "\n\n".join(context_parts)
                    
                    # Show retrieval info in sidebar if debug mode is on
                    if st.sidebar.checkbox("🔍 Show Retrieval Info"):
                        st.sidebar.write(f"Retrieved {len(docs)} relevant chunks")
                        st.sidebar.text_area("Preview of first chunk:", docs[0].page_content[:300] + "..." if docs else "No chunks")

                    # Generate answer
                    with st.spinner("🤖 Generating detailed response…"):
                        answer = chain.run(
                            context=context, 
                            question=user_q, 
                            chat_history=formatted_chat_history
                        )
                        
                        # Post-process answer to ensure it's helpful
                        if len(answer.strip()) < 50 or "I don't know" in answer or "I couldn't find" in answer:
                            # Try a more direct approach
                            direct_prompt = f"""
                            Based on the provided CBA guide context, please answer this question as specifically as possible: {user_q}
                            
                            Even if you're not completely certain, provide your best interpretation based on the available information.
                            If truly no relevant information exists, suggest related topics that might help the user.
                            """
                            answer = chain.run(
                                context=context,
                                question=direct_prompt,
                                chat_history=""
                            )
                
            except Exception as e:
                answer = f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again or rephrase your question."
                st.error(f"Error details: {e}")

    # Log & display assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=False)

# ——————————————————————————————
# 📊 Sidebar with additional features
# ——————————————————————————————
with st.sidebar:
    st.header("📋 Quick CBA Topics")
    
    # Sample questions to help users get started
    sample_questions = [
        "What is the standard discount rate for CBA?",
        "How do I calculate Net Present Value?",
        "What are the key steps in conducting a CBA?",
        "How should I handle uncertainty in my analysis?",
        "What costs should be included in a social CBA?",
        "How do I monetize benefits?",
        "What is sensitivity analysis?"
    ]
    
    st.write("💡 **Try asking:**")
    for question in sample_questions:
        if st.button(question, key=f"sample_{question}", use_container_width=True):
            # Add the question to chat
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    st.markdown("---")
    st.markdown("**💰 Cost Tip:** GPT-4o Mini provides excellent results at lower cost than GPT-4.")
