import os
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

from chatbot_utility import get_chapter_list
from get_yt_video import get_yt_video_link

load_dotenv()
DEVICE = os.getenv('DEVICE', 'cpu')

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

subjects_list = ["ML_and_DL"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScholarAI",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0a !important;
    color: #ffffff !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 720px;
    margin: auto;
}

/* ── Welcome section ── */
.welcome-container {
    text-align: center;
    padding: 5rem 0 2.5rem 0;
}
.welcome-title {
    font-size: 2.6rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
}
.welcome-subtitle {
    font-size: 2.6rem;
    color: #555555;
    letter-spacing: 0.01em;
    margin-bottom: 2rem;
}

/* ── Selectbox label ── */
label[data-testid="stWidgetLabel"] p {
    color: #888888 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* ── Selectbox dropdown ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #444 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: #141414 !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #e8e8e8 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #000000 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    background-color: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #444444 !important;
}

/* ── Markdown ── */
.stMarkdown p {
    color: #d4d4d4 !important;
    font-size: 0.95rem;
    line-height: 1.75;
}

/* ── Active course badge ── */
.course-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    color: #888;
    margin-bottom: 1.5rem;
    text-align: center;
}

/* ── Video cards ── */
.video-card {
    background: #141414;
    border: 1px solid #222222;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s ease;
}
.video-card:hover { border-color: #3a3a3a; }
.video-card-title {
    font-size: 0.88rem;
    font-weight: 500;
    color: #e0e0e0;
    margin-bottom: 0.3rem;
    line-height: 1.4;
}
.video-card-link {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #5b9cf6;
    text-decoration: none;
}
.video-section-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #444;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1rem 0 0.5rem 0;
}

/* ── Warning ── */
[data-testid="stAlert"] {
    background-color: #1a1510 !important;
    border: 1px solid #332a18 !important;
    border-radius: 8px !important;
    color: #c9a84c !important;
}

hr { border-color: #1f1f1f !important; margin: 1.2rem 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_vector_db_path(chapter, subject):
    if chapter == "All Chapters":
        return os.path.join(parent_dir, "vector_db", "machinelearning_and_deeplearning_vectordb")
    return os.path.join(parent_dir, "chapters_vector_db", chapter)


def setup_chain(selected_chapter, selected_subject):
    vector_db_path = get_vector_db_path(selected_chapter, selected_subject)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    memory = ConversationBufferMemory(output_key='answer', memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
        return_source_documents=True,
        get_chat_history=lambda h: h,
        verbose=True
    )
    return chain


def render_video_cards(video_refs):
    st.markdown('<div class="video-section-label">📺 Video References</div>', unsafe_allow_html=True)
    for title, link in video_refs:
        st.markdown(f"""
        <div class="video-card">
            <div class="video-card-title">{title}</div>
            <a class="video-card-link" href="{link}" target="_blank">▶ {link}</a>
        </div>
        """, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_history" not in st.session_state:
    st.session_state.video_history = []

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.chat_history:

    # Welcome heading
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">Welcome to ScholarAI ◈</div>
        <div class="welcome-subtitle">Welcome to Scholar AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Course + chapter selectors centered in main area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_subject = st.selectbox(
            label="Select a Course",
            options=subjects_list,
            index=None,
            placeholder="Choose a course..."
        )

        selected_chapter = None
        if selected_subject:
            chapter_list = get_chapter_list(selected_subject) + ["All Chapters"]
            selected_chapter = st.selectbox(
                label="Select a Chapter",
                options=chapter_list,
                index=0
            )

else:
    # Once chat starts, keep selectors compact at top
    selected_subject = st.session_state.get("selected_subject")
    selected_chapter = st.session_state.get("selected_chapter")

    col1, col2 = st.columns(2)
    with col1:
        selected_subject = st.selectbox(
            label="Course",
            options=subjects_list,
            index=subjects_list.index(selected_subject) if selected_subject else 0
        )
    with col2:
        if selected_subject:
            chapter_list = get_chapter_list(selected_subject) + ["All Chapters"]
            ch_index = chapter_list.index(selected_chapter) if selected_chapter in chapter_list else 0
            selected_chapter = st.selectbox(
                label="Chapter",
                options=chapter_list,
                index=ch_index
            )

    st.markdown("<hr>", unsafe_allow_html=True)

# ── Persist selections ────────────────────────────────────────────────────────
if selected_subject:
    st.session_state.selected_subject = selected_subject
if selected_chapter:
    if st.session_state.get('selected_chapter') != selected_chapter:
        st.session_state.chat_chain = setup_chain(selected_chapter, selected_subject)
    st.session_state.selected_chapter = selected_chapter

# ── Chat history ──────────────────────────────────────────────────────────────
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and idx < len(st.session_state.video_history):
            video_refs = st.session_state.video_history[idx]
            if video_refs:
                render_video_cards(video_refs)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your course...")

if user_input:
    if "chat_chain" not in st.session_state:
        st.warning("Please select a course and chapter first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.video_history.append(None)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_chain({"question": user_input})
            st.markdown(response['answer'])

            search_query = ', '.join([
                item["content"] for item in st.session_state.chat_history
                if item["role"] == "user"
            ])
            video_titles, video_links = get_yt_video_link(search_query)

            video_refs = []
            if video_titles:
                for i in range(min(3, len(video_titles))):
                    video_refs.append((video_titles[i], video_links[i]))
                render_video_cards(video_refs)

            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
            st.session_state.video_history.append(video_refs)