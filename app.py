import os
import requests
import streamlit as st
import pandas as pd
from config import config
from asr import transcribe_audio
from extractor import extract_tasks
from export import export_tasks_to_csv, export_report_to_pdf
from create_sample_audio import SAMPLE_PRESETS, generate_sample_wav_file

# Page Config
st.set_page_config(
    page_title="VoxTask - Voice-First Multilingual Task Extractor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #9333EA, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-med {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .segment-box {
        background-color: #FFFFFF;
        border-left: 4px solid #6366F1;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "segments" not in st.session_state:
    st.session_state.segments = []
if "tasks" not in st.session_state:
    st.session_state.tasks = []
if "backend_used" not in st.session_state:
    st.session_state.backend_used = "Not run yet"
if "meeting_title" not in st.session_state:
    st.session_state.meeting_title = "Sprint Planning Sync"

def check_ollama_status(host: str) -> bool:
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ VoxTask Engine Config")

# ASR Model Selector
st.sidebar.subheader("🎙️ ASR Models")
asr_routing = st.sidebar.radio(
    "Routing Logic",
    ["Hybrid (Whisper + Custom Swecha ASR)", "Force Custom Telugu Model (swecha-gonthuka-asr)", "Whisper Only"],
    index=0
)

# LLM Backend Selector
st.sidebar.subheader("🧠 LLM Extractor Backend")
ollama_active = check_ollama_status(config.OLLAMA_HOST)

if ollama_active:
    st.sidebar.success(f"🟢 Ollama Connected ({config.OLLAMA_HOST})")
else:
    st.sidebar.warning("🟡 Ollama Offline (Using Fallback/BYOK)")

llm_backend = st.sidebar.selectbox(
    "Task Extractor Provider",
    ["ollama", "openai", "anthropic", "regex"],
    index=0 if ollama_active else 3,
    format_func=lambda x: {
        "ollama": "Local Ollama (Mistral)",
        "openai": "OpenAI (Cloud BYOK)",
        "anthropic": "Anthropic Claude (Cloud BYOK)",
        "regex": "Rule-Based Parser (Local Offline)"
    }[x]
)

byok_key = ""
if llm_backend in ["openai", "anthropic"]:
    byok_key = st.sidebar.text_input(
        f"Enter {llm_backend.upper()} API Key",
        type="password",
        value=config.OPENAI_API_KEY if llm_backend == "openai" else config.ANTHROPIC_API_KEY
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🎭 Quick Demo Presets")
st.sidebar.caption("Instant 1-click loading for hackathon stage rehearsal")

preset_choice = st.sidebar.selectbox(
    "Load Pre-Recorded Sample",
    ["-- Select Sample Preset --"] + list(SAMPLE_PRESETS.keys())
)

if preset_choice != "-- Select Sample Preset --":
    if st.sidebar.button("⚡ Load Preset Data"):
        preset_info = SAMPLE_PRESETS[preset_choice]
        st.session_state.segments = preset_info["segments"]
        st.session_state.meeting_title = preset_choice
        
        # Auto extract tasks for instant demo
        force_telugu = (asr_routing == "Force Custom Telugu Model (swecha-gonthuka-asr)")
        extracted_tasks, b_used = extract_tasks(
            st.session_state.segments,
            backend=llm_backend,
            api_key=byok_key
        )
        st.session_state.tasks = extracted_tasks
        st.session_state.backend_used = b_used
        st.sidebar.success(f"Loaded '{preset_choice}'!")

# Header Section
st.markdown('<div class="main-title">🎙️ VoxTask: Multilingual Voice-to-Task Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Fine-tuned Telugu ASR (Swecha-Gonthuka) + Whisper-small + Local LLM Task Extraction</div>', unsafe_allow_html=True)

# Top Info Bar
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"**Primary ASR:** `Whisper-small`")
with c2:
    st.markdown(f"**Telugu ASR:** `swecha-gonthuka`")
with c3:
    st.markdown(f"**Extractor:** `{st.session_state.backend_used}`")
with c4:
    st.markdown(f"**Status:** `Demo Ready` 🚀")

st.markdown("---")

# Main Content Layout
col_audio, col_dash = st.columns([1, 2])

with col_audio:
    st.subheader("1. Audio Input")
    
    input_method = st.radio("Choose Input Source:", ["Upload Audio File", "Live Microphone Recording"])
    
    audio_bytes = None
    if input_method == "Upload Audio File":
        uploaded_file = st.file_uploader("Upload Meeting Recording (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a", "ogg"])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format="audio/wav")
    else:
        # Live Microphone Input
        mic_audio = st.audio_input("Record Voice Memo / Live Meeting")
        if mic_audio is not None:
            audio_bytes = mic_audio.read()

    # Generate synthetic wav sample button if user has no file ready
    if st.button("🎵 Generate Demo Audio WAV"):
        sample_path = generate_sample_wav_file()
        with open(sample_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")
        st.success("Generated sample 16kHz WAV file!")

    st.markdown("---")
    meeting_title_input = st.text_input("Meeting Title / Topic", value=st.session_state.meeting_title)
    st.session_state.meeting_title = meeting_title_input

    process_button = st.button("🚀 Transcribe & Extract Tasks", type="primary", use_container_width=True)

    if process_button:
        if audio_bytes is None and not st.session_state.segments:
            st.warning("Please upload an audio file, record audio, or select a Demo Preset.")
        else:
            with st.spinner("Step 1/2: Running Multilingual ASR (Whisper + Swecha Telugu)..."):
                force_telugu = (asr_routing == "Force Custom Telugu Model (swecha-gonthuka-asr)")
                if audio_bytes is not None:
                    segments = transcribe_audio(audio_bytes, force_telugu_model=force_telugu)
                    st.session_state.segments = segments

            with st.spinner(f"Step 2/2: Extracting Action Items using {llm_backend.upper()}..."):
                tasks, backend_used = extract_tasks(
                    st.session_state.segments,
                    backend=llm_backend,
                    api_key=byok_key
                )
                st.session_state.tasks = tasks
                st.session_state.backend_used = backend_used

            st.success("Extraction Complete!")
            st.balloons()

# Dashboard Output
with col_dash:
    tab_tasks, tab_transcript, tab_export = st.tabs(["📋 Extracted Tasks", "🗣️ Full Transcript", "📊 Analytics & Export"])

    with tab_tasks:
        st.subheader("Extracted Action Items")
        tasks = st.session_state.tasks

        if not tasks:
            st.info("No tasks extracted yet. Upload audio or select a preset on the left.")
        else:
            # Summary Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            high_count = sum(1 for t in tasks if t.get("priority", "").lower() == "high")
            unassigned_count = sum(1 for t in tasks if t.get("owner", "").lower() in ["unassigned", "none", ""])
            
            with m1:
                st.metric("Total Tasks", len(tasks))
            with m2:
                st.metric("High Priority", high_count, delta="Urgent" if high_count > 0 else None, delta_color="inverse")
            with m3:
                st.metric("Unassigned", unassigned_count)
            with m4:
                st.metric("Backend Engine", st.session_state.backend_used.split()[0])

            st.markdown("#### Task Checklist")
            for idx, task in enumerate(tasks):
                priority = task.get("priority", "Medium").title()
                badge_class = "badge-high" if priority == "High" else ("badge-med" if priority == "Medium" else "badge-low")
                
                with st.container():
                    c_chk, c_desc, c_owner, c_due, c_prio = st.columns([0.5, 3.5, 1.5, 1.5, 1.0])
                    with c_chk:
                        st.checkbox("", key=f"chk_{idx}")
                    with c_desc:
                        st.markdown(f"**{task.get('action')}**")
                        if task.get("context_snippet"):
                            st.caption(f"💬 *\"{task.get('context_snippet')}\"*")
                    with c_owner:
                        st.markdown(f"👤 `{task.get('owner')}`")
                    with c_due:
                        st.markdown(f"📅 `{task.get('deadline')}`")
                    with c_prio:
                        st.markdown(f'<span class="{badge_class}">{priority}</span>', unsafe_allow_html=True)
                    st.divider()

            with st.expander("🔍 View Raw JSON Output"):
                st.json(tasks)

    with tab_transcript:
        st.subheader("Multilingual Transcript with Speaker Tags")
        segments = st.session_state.segments
        
        if not segments:
            st.info("Transcript will appear here after processing.")
        else:
            for seg in segments:
                speaker = seg.get("speaker", "Speaker")
                lang = seg.get("language", "English")
                engine = seg.get("engine", "ASR")
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                text = seg.get("text", "")
                
                st.markdown(f"""
                <div class="segment-box">
                    <div style="display: flex; justify-space-between; align-items: center; margin-bottom: 6px;">
                        <strong>⏱️ {start}s - {end}s | {speaker}</strong>
                        <span style="background: #EEF2FF; color: #4338CA; padding: 2px 8px; border-radius: 8px; font-size: 0.75rem;">{lang} • {engine}</span>
                    </div>
                    <div style="font-size: 1.0rem; color: #1F2937;">{text}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab_export:
        st.subheader("📊 Analytics & Report Export")
        tasks = st.session_state.tasks
        segments = st.session_state.segments

        if tasks:
            df = pd.DataFrame(tasks)
            st.markdown("#### Task Priority Distribution")
            prio_counts = df["priority"].value_counts()
            st.bar_chart(prio_counts)

            st.markdown("#### Export Options")
            col_pdf, col_csv = st.columns(2)
            
            with col_pdf:
                pdf_bytes = export_report_to_pdf(st.session_state.meeting_title, segments, tasks)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"VoxTask_{st.session_state.meeting_title.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
            with col_csv:
                csv_bytes = export_tasks_to_csv(tasks)
                st.download_button(
                    label="📊 Download Tasks CSV",
                    data=csv_bytes,
                    file_name=f"VoxTask_{st.session_state.meeting_title.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("Process an audio recording or select a preset to generate export reports.")
