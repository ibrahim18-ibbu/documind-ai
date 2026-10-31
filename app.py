import hashlib
import streamlit as st
import fitz
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer


APP_TITLE = "DocuMind AI"
OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 1000, 150, 5

st.set_page_config(page_title=APP_TITLE, page_icon="✦", layout="wide")

st.markdown("""
<style>
/* ==========================================
   DOCUMIND AI — HIGH CONTRAST COLOUR FIX
   ========================================== */

/* Main application background */
.stApp,
[data-testid="stAppViewContainer"] {
    background: #F3F4FF !important;
    color: #172554 !important;
}

/* All normal text */
.stApp p,
.stApp li,
.stApp label,
.stApp span,
.stApp div[data-testid="stMarkdownContainer"] {
    color: #172554;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #1E1B4B !important;
    font-weight: 800 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #11113D 0%,
        #21105A 55%,
        #321078 100%
    ) !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
    color: #F8FAFF !important;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* File uploader area */
[data-testid="stFileUploaderDropzone"] {
    background: #EEF2FF !important;
    border: 2px dashed #818CF8 !important;
    border-radius: 14px !important;
}

/* File uploader text */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span {
    color: #312E81 !important;
}

/* Sidebar uploader button */
section[data-testid="stSidebar"]
[data-testid="stFileUploaderDropzone"] button {
    background: #FFFFFF !important;
    color: #312E81 !important;
    border: 1px solid #A5B4FC !important;
}

/* Normal buttons */
.stButton > button {
    background: #FFFFFF !important;
    color: #3730A3 !important;
    border: 1px solid #A5B4FC !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #7C3AED,
        #4F46E5
    ) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 5px 14px #6366F155 !important;
}

/* Button text must remain visible */
.stButton > button *,
.stButton > button[kind="primary"] * {
    color: inherit !important;
}

/* Hover effect */
.stButton > button:hover {
    border-color: #7C3AED !important;
    transform: translateY(-1px);
}

/* Metric cards */
.metric {
    background: #FFFFFF !important;
    border: 1px solid #D8D9FF !important;
    border-radius: 18px !important;
    box-shadow: 0 5px 18px #4338CA12 !important;
}

.metric-label {
    color: #4338CA !important;
    font-weight: 800 !important;
}

.metric-value {
    color: #6D28D9 !important;
    font-weight: 900 !important;
}

.metric-note {
    color: #475569 !important;
}

/* Main content panels */
.panel {
    background: #FFFFFF !important;
    color: #172554 !important;
    border: 1px solid #D8D9FF !important;
    border-radius: 18px !important;
}

/* Chat message cards */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    color: #172554 !important;
    border: 1px solid #D8D9FF !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 14px #4338CA10 !important;
}

/* Chat text: prevent pale or invisible text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em {
    color: #172554 !important;
    -webkit-text-fill-color: #172554 !important;
}

/* Chat input container */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 2px solid #8B5CF6 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 16px #7C3AED18 !important;
}

/* Chat input text */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    background: #FFFFFF !important;
    color: #172554 !important;
    -webkit-text-fill-color: #172554 !important;
    caret-color: #6D28D9 !important;
    opacity: 1 !important;
}

/* Placeholder text */
[data-testid="stChatInput"] textarea::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
    opacity: 1 !important;
}

/* Chat send button */
[data-testid="stChatInput"] button {
    background: linear-gradient(
        135deg,
        #7C3AED,
        #4F46E5
    ) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}

/* Chat send button icon */
[data-testid="stChatInput"] button svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* Text inputs and text areas */
.stTextInput input,
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #172554 !important;
    -webkit-text-fill-color: #172554 !important;
    border: 1px solid #A5B4FC !important;
}

/* Expandable source sections */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #C7D2FE !important;
    border-radius: 14px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {
    color: #3730A3 !important;
}

/* Information and tip boxes */
.tip {
    background: #E0F2FE !important;
    color: #075985 !important;
    border: 1px solid #7DD3FC !important;
    border-radius: 14px !important;
}

.tip b {
    color: #075985 !important;
}

/* Success messages */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* Links */
.stApp a {
    color: #6D28D9 !important;
    font-weight: 700;
}

/* Divider */
hr {
    border-color: #D8D9FF !important;
}

/* Footer */
.footer {
    color: #64748B !important;
}

/* Mobile layout */
@media (max-width: 700px) {
    .block-container {
        padding: 1rem !important;
    }

    .hero {
        padding: 22px !important;
    }

    .metric {
        padding: 12px !important;
    }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

def ollama_ready():
    try:
        ollama.list()
        return True, None
    except Exception as e:
        return False, str(e)

def extract_pages(files):
    pages = []
    for f in files:
        with fitz.open(stream=f.getvalue(), filetype="pdf") as pdf:
            for number, page in enumerate(pdf, 1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"text": text, "source": f.name, "page": number})
    return pages

def chunk_pages(pages):
    chunks = []
    for p in pages:
        start, text = 0, p["text"]
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            section = text[start:end].strip()
            if section:
                chunks.append({"text": section, "source": p["source"], "page": p["page"]})
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP
    return chunks

def embed_chunks(chunks, model):
    return np.asarray(model.encode([c["text"] for c in chunks],
        normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

def retrieve(question, chunks, vectors, model):
    q = np.asarray(model.encode([question], normalize_embeddings=True,
        show_progress_bar=False), dtype=np.float32)[0]
    scores = vectors @ q
    ids = np.argsort(scores)[::-1][:min(TOP_K, len(chunks))]
    return [{**chunks[int(i)], "score": float(scores[int(i)])} for i in ids]

def generate_answer(question, sources, history):
    context = "\n\n---\n\n".join(
        f"[Source: {s['source']}, page {s['page']}]\n{s['text']}" for s in sources)
    recent = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
    prompt = f"""You are a helpful assistant for PDFs on any subject.
Use retrieved PDF excerpts as the main evidence for document-specific facts.
Answer clearly and directly. Explain, summarize, compare, or calculate when supported.
Cite document facts using the exact filename and page number in the excerpts.
If the excerpts do not provide enough information, say you could not find enough information in the uploaded PDFs.
Do not invent facts or claim a PDF says something unsupported. Label any added general knowledge clearly.

Recent conversation:
{recent if recent else "(No previous conversation.)"}

Retrieved PDF excerpts:
{context}

Current question:
{question}

Answer:"""
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def clear_docs():
    for key in ("chunks", "embeddings", "processed_files", "messages"):
        st.session_state.pop(key, None)

with st.sidebar:
    st.markdown("""<div class="brand"><div class="brand-icon">✦</div><div><div class="brand-title">DocuMind AI</div><div class="brand-sub">UNIVERSAL PDF ASSISTANT</div></div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Your knowledge base</div>', unsafe_allow_html=True)
    files = st.file_uploader("Add PDF documents", type=["pdf"], accept_multiple_files=True)
    process = st.button("✦  Process documents", type="primary", use_container_width=True)
    if st.session_state.get("processed_files"):
        st.markdown('<div class="sidebar-label">Indexed documents</div>', unsafe_allow_html=True)
        for filename in st.session_state.processed_files:
            st.markdown(f"📄 {filename}")
        st.caption(f"{len(st.session_state.processed_files)} document(s) ready")
    st.markdown('<div class="sidebar-label">Actions</div>', unsafe_allow_html=True)
    if st.button("＋  Start a new chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("Clear documents & chat", use_container_width=True):
        clear_docs()
        st.rerun()
    st.markdown("---")
    st.markdown('<div class="sidebar-label">Local AI engine</div>', unsafe_allow_html=True)
    st.markdown(f"**Model**  \n`{OLLAMA_MODEL}`")
    st.markdown("🟢 **No API key required**")
    st.caption("Answers are generated by Ollama on your computer.")

for key, value in {"messages": [], "chunks": [], "embeddings": np.empty((0, 0), dtype=np.float32), "processed_files": []}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""<div class="hero"><div class="eyebrow">YOUR DOCUMENTS. YOUR ANSWERS.</div><h1>Make every PDF<br>easy to understand.</h1><p>Upload documents, ask questions in plain language, and get useful answers grounded in your files — powered by local AI.</p><div class="pills"><span class="pill">✦ Any subject</span><span class="pill">⌕ Semantic search</span><span class="pill">🔒 Local Ollama AI</span><span class="pill">↗ Page-level sources</span></div></div>""", unsafe_allow_html=True)

if process:
    if not files:
        st.warning("Upload at least one PDF first.")
    else:
        try:
            ready, err = ollama_ready()
            if not ready:
                st.error("Ollama is not reachable. Open Ollama and make sure the model is installed.")
                st.code("ollama run llama3.2:3b")
                st.caption(str(err))
            else:
                with st.spinner("Reading your PDF documents..."):
                    pages = extract_pages(files)
                if not pages:
                    st.error("No selectable text found. Scanned PDFs need OCR, not included in this version.")
                else:
                    chunks = chunk_pages(pages)
                    with st.spinner("Building the local document index..."):
                        embedder = load_embedder()
                        vectors = embed_chunks(chunks, embedder)
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = vectors
                    st.session_state.processed_files = [f.name for f in files]
                    st.session_state.messages = []
                    st.success(f"Indexed {len(files)} PDF(s), {len(pages)} text pages and {len(chunks)} searchable sections.")
                    st.rerun()
        except Exception as e:
            st.error(f"Could not process documents: {e}")

ready = bool(st.session_state.chunks)
doc_count = len(st.session_state.processed_files)
chunk_count = len(st.session_state.chunks)
question_count = sum(m["role"] == "user" for m in st.session_state.messages)
c1, c2, c3 = st.columns(3)
for col, label, value, note in [
    (c1, "DOCUMENTS INDEXED", doc_count, "PDF files in this session"),
    (c2, "SEARCHABLE SECTIONS", f"{chunk_count:,}", "Text segments ready to search"),
    (c3, "QUESTIONS ASKED", question_count, "Questions in this chat")
]:
    with col:
        st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Ask your documents</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Request a summary, explore a concept, or ask a question about your PDFs.</div>', unsafe_allow_html=True)

if not ready:
    a, b, c = st.columns(3)
    for col, icon, title, desc in [
        (a, "📄", "01 · Upload", "Add one or more text-based PDFs from the sidebar."),
        (b, "⌕", "02 · Index", "The app extracts text and builds a local search index."),
        (c, "✦", "03 · Ask", "Ask questions and inspect the sources behind answers.")
    ]:
        with col:
            st.markdown(f'<div class="panel"><div style="font-size:24px">{icon}</div><b>{title}</b><p style="font-size:13px;color:#718096">{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📖 Sources used for this answer"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']} · Page {s['page']}** · Similarity `{s['score']:.3f}`")
                    st.write(s["text"])
                    st.divider()

if ready and not st.session_state.messages:
    st.markdown('<div class="tip">💡 <b>Try asking:</b> “Summarize this document”, “Explain the key concepts”, or “What does the PDF say about [topic]?”</div>', unsafe_allow_html=True)

question = st.chat_input("Ask a question about your PDFs..." if ready else "Upload and process a PDF to start chatting")
if question:
    if not ready:
        st.warning("Upload and process PDFs before asking questions.")
    else:
        history = st.session_state.messages.copy()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching relevant passages..."):
                    embedder = load_embedder()
                    sources = retrieve(question, st.session_state.chunks, st.session_state.embeddings, embedder)
                with st.spinner("Generating answer with local AI..."):
                    answer = generate_answer(question, sources, history)
                st.markdown(answer)
                with st.expander("📖 Sources used for this answer"):
                    for s in sources:
                        st.markdown(f"**{s['source']} · Page {s['page']}** · Similarity `{s['score']:.3f}`")
                        st.write(s["text"])
                        st.divider()
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as e:
                st.error("Could not generate an answer. Check that Ollama is running and the model is installed.")
                st.code("ollama run llama3.2:3b")
                st.caption(str(e))

st.markdown('<div class="footer">DOCUMIND AI · Streamlit + Sentence Transformers + Ollama · No cloud API key required</div>', unsafe_allow_html=True)
