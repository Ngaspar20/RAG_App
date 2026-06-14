import streamlit as st
import os
import tempfile
import uuid

from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    # API key — reads from Streamlit secrets if available, otherwise shows input
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ API key loaded from secrets")
    except Exception:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at console.groq.com",
        )

    st.divider()

    model_choice = st.selectbox(
        "LLM Model",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        index=0,
        help="Larger models give better answers but may be slightly slower.",
    )

    k_chunks = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=5,
        value=3,
        help="How many passages from the document to use as context per question.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 = consistent and precise. Higher values = more creative but less predictable.",
    )

    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Upload any PDF")
    st.markdown("2. App chunks & indexes it locally")
    st.markdown("3. Ask questions in natural language")
    st.markdown("4. Answers are grounded in your document with page citations")
    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq · Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# Cached resources — loaded once per server session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (first time only)…")
def load_embedding_model():
    """Load all-MiniLM-L6-v2 once and cache it for all users."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Document processing helpers
# ─────────────────────────────────────────────────────────────────────────────
def process_document(uploaded_file) -> list:
    """
    Save the uploaded BytesIO to a temp file, load pages with PyMuPDF,
    split into 1000-token chunks with 100-token overlap.
    Returns a list of LangChain Document objects.
    """
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=1000,
            chunk_overlap=100,
        )
        chunks = loader.load_and_split(splitter)
    finally:
        os.unlink(tmp_path)  # always clean up the temp file

    return chunks


def build_vectorstore(chunks: list, embedding_model) -> Chroma:
    """
    Embed all chunks and store them in an in-memory ChromaDB collection.
    A unique collection name is used so re-indexing always starts fresh.
    """
    collection_name = f"rag_{uuid.uuid4().hex[:8]}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
    )
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# RAG generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_rag_response(
    question: str,
    vectorstore: Chroma,
    k: int,
    api_key: str,
    model: str,
    temperature: float,
) -> tuple[str, list]:
    """
    1. Retrieve top-k chunks from ChromaDB via cosine similarity.
    2. Build a structured prompt with the retrieved context.
    3. Call Groq API and return (answer, source_docs).
    """
    # ── Retrieval ──────────────────────────────────────────────────────────
    source_docs = vectorstore.similarity_search(question, k=k)

    context_parts = []
    for doc in source_docs:
        page_num = doc.metadata.get("page", 0) + 1  # PyMuPDF uses 0-based pages
        context_parts.append(f"[Page {page_num}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # ── Generation ─────────────────────────────────────────────────────────
    client = Groq(api_key=api_key)

    system_message = (
        "You are a precise document assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "Always cite the page number(s) your answer is drawn from, like (Page 12). "
        "If the answer cannot be found in the context, say clearly: "
        "'This information is not available in the uploaded document.' "
        "Do not guess or add information from outside the context."
    )

    user_message = (
        f"Context extracted from the document:\n\n{context}\n\n"
        f"Question: {question}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=512,
    )

    answer = response.choices[0].message.content
    return answer, source_docs


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {question, answer, sources}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📄 RAG Document Assistant")
st.markdown(
    "Upload any PDF and ask questions about it. "
    "Every answer is grounded in your document and cites the source page."
)

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Supported: any PDF up to 200 MB.",
)

if uploaded_file:
    # ── Detect new document and (re)index ─────────────────────────────────
    if st.session_state.doc_name != uploaded_file.name:
        st.session_state.chat_history = []   # clear chat on new document

        with st.status("Indexing document…", expanded=True) as status_box:
            st.write("📄 Reading PDF pages…")
            embedding_model = load_embedding_model()

            st.write("✂️ Splitting into chunks…")
            chunks = process_document(uploaded_file)

            st.write(f"🔢 Embedding {len(chunks)} chunks into ChromaDB…")
            vectorstore = build_vectorstore(chunks, embedding_model)

            st.session_state.vectorstore = vectorstore
            st.session_state.doc_name = uploaded_file.name
            st.session_state.num_chunks = len(chunks)

            status_box.update(label="✅ Document ready!", state="complete", expanded=False)

        st.success(
            f"**{uploaded_file.name}** indexed — "
            f"{st.session_state.num_chunks} chunks created. Ask your first question below."
        )

    else:
        st.info(
            f"**{uploaded_file.name}** is already indexed "
            f"({st.session_state.num_chunks} chunks). Ask a question below."
        )

    st.divider()

    # ── Render chat history ───────────────────────────────────────────────
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            with st.expander("📖 Source passages used"):
                for i, doc in enumerate(entry["sources"]):
                    page_num = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Page {page_num}**")
                    st.caption(doc.page_content[:500] + ("…" if len(doc.page_content) > 500 else ""))
                    if i < len(entry["sources"]) - 1:
                        st.divider()

    # ── Chat input ────────────────────────────────────────────────────────
    question = st.chat_input("Ask a question about your document…")

    if question:
        if not api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar to continue.")
            st.stop()

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer…"):
                try:
                    answer, source_docs = generate_rag_response(
                        question=question,
                        vectorstore=st.session_state.vectorstore,
                        k=k_chunks,
                        api_key=api_key,
                        model=model_choice,
                        temperature=temperature,
                    )

                    st.markdown(answer)

                    with st.expander("📖 Source passages used"):
                        for i, doc in enumerate(source_docs):
                            page_num = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**Page {page_num}**")
                            st.caption(doc.page_content[:500] + ("…" if len(doc.page_content) > 500 else ""))
                            if i < len(source_docs) - 1:
                                st.divider()

                    # Persist to session history
                    st.session_state.chat_history.append(
                        {"question": question, "answer": answer, "sources": source_docs}
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info(
                        "Common causes: invalid API key, Groq rate limit reached, "
                        "or the prompt exceeded the model's context window. "
                        "Try reducing the 'Chunks to retrieve' slider."
                    )

else:
    # ── Empty state — shown before any file is uploaded ───────────────────
    st.info("👆 Upload a PDF to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Semantic Search")
        st.markdown(
            "Finds relevant passages by **meaning**, not just keywords. "
            "Ask in natural language."
        )
    with col2:
        st.markdown("### 📍 Page Citations")
        st.markdown(
            "Every answer shows **which page** it came from, "
            "so you can verify in the original document."
        )
    with col3:
        st.markdown("### 💬 Conversation History")
        st.markdown(
            "Ask **multiple questions** in the same session. "
            "Upload a new file to start fresh."
        )
