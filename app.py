import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

st.set_page_config(
    page_title="AURORA | PowerTech Solutions",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {

    background:

    radial-gradient(
    circle at top right,
    rgba(0,180,216,.10),
    transparent 35%
    ),

    linear-gradient(
    180deg,
    #020812,
    #05101d,
    #071523
    );

    color:#c8d8f0;
    }

    /* 1. Remove o fundo do container mais externo */
    div[data-testid="stChatInput"] {
        background: transparent !important;
    }

    /* 2. Aplica a cor e a borda no container principal que envolve tudo */
    div[data-testid="stChatInput"] > div {
        border: 1px solid rgba(0, 180, 216, 0.3) !important;
        background-color: #0d1f35 !important; /* Cor sólida e uniforme */
        border-radius: 15px !important;
        transition: all 0.3s ease;
    }

    /* 3. O SEGREDO AQUI: Força a transparência em TODAS as sub-camadas (Base Web) */
    div[data-testid="stChatInput"] div[data-baseweb="base-input"],
    div[data-testid="stChatInput"] div[data-baseweb="textarea"],
    div[data-testid="stChatInput"] div[data-baseweb="textarea"] > div,
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important; /* Remove qualquer borda interna residual */
        box-shadow: none !important;
        color: #c8d8f0 !important;
    }

    /* 4. Efeito ao clicar (Focus) com o brilho neon */
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #00b4d8 !important;
        box-shadow: 0 0 0 1px #00b4d8 inset, 0 0 15px rgba(0, 180, 216, 0.15) !important; 
    }

    /* Estiliza o botão de enviar (seta) quando você digita algo */
    div[data-testid="stChatInput"] button {
        background-color: #00b4d8 !important; /* Fundo azul neon */
        color: #05101d !important; /* Seta escura para dar contraste */
        transition: all 0.3s ease;
    }

    /* Efeito de brilho quando passar o mouse por cima da seta */
    div[data-testid="stChatInput"] button:hover {
        background-color: #0096b4 !important; 
        box-shadow: 0 0 10px rgba(0, 180, 216, 0.6) !important;
    }

    /* Estilo do botão apagado quando a caixa de texto está vazia */
    div[data-testid="stChatInput"] button[disabled] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: rgba(255, 255, 255, 0.2) !important;
        box-shadow: none !important;
    }
    
    /* Estilos da Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #071020 100%) !important;
        border-right: 1px solid #1a3050 !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1rem 1rem 3rem 1rem !important;
        overflow-y: auto !important;
    }
    [data-testid="stSidebar"] * { color: #c8d8f0 !important; }

    .sidebar-logo {
        text-align: center;
        padding: 16px 0 24px 0;
        border-bottom: 1px solid #1a3050;
        margin-bottom: 20px;
    }
    .sidebar-logo .icon { font-size: 2.8rem; display: block; margin-bottom: 6px; }
    .sidebar-logo h2 {
        font-family: 'JetBrains Mono', monospace;
        color: #00b4d8 !important;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 2px;
    }
    .sidebar-logo p {
        color: #4a6fa5 !important;
        font-size: 0.65rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 4px 0 0 0;
    }

    .info-card {
        background: #0d1f35;
        border: 1px solid #1a3050;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .info-card .label {
        font-size: 0.6rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #4a6fa5 !important;
        margin-bottom: 3px;
    }
    .info-card .value {
        font-size: 0.82rem;
        font-weight: 600;
        color: #c8d8f0 !important;
    }

    .status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 0;
        font-size: 0.78rem;
        color: #c8d8f0 !important;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .dot-green { background: #00e5a0; box-shadow: 0 0 6px #00e5a0; }
    .dot-blue  { background: #00b4d8; box-shadow: 0 0 6px #00b4d8; }
    .dot-gray  { background: #4a6fa5; }

    .sidebar-divider { border: none; border-top: 1px solid #1a3050; margin: 16px 0; }

    .hero-container{

        display:flex;

        justify-content:space-between;

        align-items:center;

        padding:30px;

        margin-bottom:20px;

        border-radius:22px;

        background:
        linear-gradient(
        135deg,
        #07111f,
        #09213a,
        #0b3d5d
        );

        border:
        1px solid rgba(0,180,216,.25);

        box-shadow:
        0 0 40px rgba(0,180,216,.08);
        }

        .hero-left{

        display:flex;

        align-items:center;

        gap:22px;
        }

        .hero-logo{

        font-size:70px;
        }

        .hero-title{

        font-size:28px;

        font-weight:700;

        font-family:'JetBrains Mono', monospace;

        letter-spacing:2px;

        color:white;
        }

        .hero-subtitle{

        margin-top:8px;

        font-size:18px;

        line-height:1.6;

        color:#7fc8ff;

        max-width:800px;
        }

        .hero-badges{

        margin-top:16px;
        }

        .hero-badges span{

        display:inline-block;

        margin-right:10px;

        padding:8px 14px;

        border-radius:30px;

        background:
        rgba(0,180,216,.12);

        border:
        1px solid rgba(0,180,216,.25);

        color:#00d4ff;

        font-size:12px;

        font-weight:600;
        }

        .hero-status{

        display:flex;

        align-items:center;

        gap:15px;

        padding:18px 22px;

        border-radius:18px;

        background:
        rgba(0,229,160,.08);

        border:
        1px solid rgba(0,229,160,.25);

        min-width:280px;
        }

        .status-dot-big{

        width:14px;

        height:14px;

        border-radius:50%;

        background:#00e676;

        animation:pulse 2s infinite;

        box-shadow:
        0 0 10px #00e676;
        }

        .status-title{

        font-size:14px;

        font-weight:700;

        color:white;
        }

        .status-time{

        font-size:13px;

        color:#8cd9b4;

        margin-top:4px;
        }

        @keyframes pulse {

        0%{
        box-shadow:0 0 0 0 rgba(0,230,118,.7);
        }

        70%{
        box-shadow:0 0 0 15px rgba(0,230,118,0);
        }

        100%{
        box-shadow:0 0 0 0 rgba(0,230,118,0);
        }

        }
    
    /* Conteúdo principal */
    .main-header {
        background: linear-gradient(135deg, #0a1628 0%, #0d1f35 60%, #071a2e 100%);
        border: 1px solid #1a3050;
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.7rem;
        font-weight: 700;
        color: #00b4d8;
        margin: 0 0 6px 0;
    }
    .main-header p { color: #4a6fa5; margin: 0; font-size: 0.88rem; }
    .tag {
        display: inline-block;
        background: rgba(0,180,216,0.15);
        border: 1px solid rgba(0,180,216,0.3);
        color: #00b4d8;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 20px;
        text-transform: uppercase;
        margin-top: 10px;
        margin-right: 6px;
    }
    .tag-cold { background: rgba(0,229,160,0.1); border-color: rgba(0,229,160,0.25); color: #00e5a0; }

    .stChatMessage {

    background:
    rgba(13,31,53,.88) !important;

    backdrop-filter:
    blur(10px);

    border:
    1px solid rgba(0,180,216,.15) !important;

    border-radius:20px !important;

    padding:12px !important;

    transition:.3s;
    }

    .stChatMessage:hover{

    border-color:#00b4d8 !important;

    box-shadow:
    0 0 20px rgba(0,180,216,.08);
    }

    .stButton > button {
        background: transparent !important;
        border: 1px solid #1a3050 !important;
        color: #4a6fa5 !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        border-color: #00b4d8 !important;
        color: #00b4d8 !important;
        background: rgba(0,180,216,0.06) !important;
    }

    .stSpinner > div { color: #00b4d8 !important; }
    
    /* CORREÇÃO: O 'header' foi removido desta lista para permitir que o botão de abrir a sidebar fique visível */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── API Key ───────────────────────────────────────────────────────────────────
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
try:
    if "NVIDIA_API_KEY" in st.secrets:
        nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
except Exception:
    pass

pdf_ok = os.path.exists("manual.pdf")
api_ok = bool(nvidia_api_key)

# ── SIDEBAR ── renderizada ANTES de qualquer st.stop() ───────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="icon">❄️</span>
        <h2>AURORA AI</h2>
        <p>PowerTech Solutions</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="label">Produto</div>
        <div class="value">🏭 Sistema de Refrigeração Industrial</div>
    </div>
    <div class="info-card">
        <div class="label">Equipe Responsável</div>
        <div class="value">🏢 PowerTech Solutions</div>
    </div>
    <div class="info-card">
        <div class="label">Modelo de IA</div>
        <div class="value">🤖 meta/llama-3.1-8b-instruct</div>
    </div>
    <div class="info-card">
        <div class="label">Embeddings</div>
        <div class="value">🔢 nvidia/nv-embedqa-e5-v5</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.65rem; letter-spacing:1.5px; text-transform:uppercase; color:#4a6fa5; margin-bottom:8px;">Status do Sistema</div>
    <div class="status-row">
        <div class="dot {'dot-green' if pdf_ok else 'dot-gray'}"></div>
        <span>Manual PDF {'carregado' if pdf_ok else 'não encontrado'}</span>
    </div>
    <div class="status-row">
        <div class="dot {'dot-green' if api_ok else 'dot-gray'}"></div>
        <span>API Key {'configurada' if api_ok else 'ausente'}</span>
    </div>
    <div class="status-row">
        <div class="dot {'dot-blue' if (pdf_ok and api_ok) else 'dot-gray'}"></div>
        <span>Vetorização FAISS {'ativa' if (pdf_ok and api_ok) else 'aguardando'}</span>
    </div>
    <div class="status-row">
        <div class="dot {'dot-green' if api_ok else 'dot-gray'}"></div>
        <span>IA {'conectada' if api_ok else 'desconectada'}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    if st.button("🗑️  Limpar Conversa"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Conversa reiniciada. Como posso ajudar? ❄️"}
        ]
        st.rerun()

    st.markdown("""
    <div style="margin-top:20px; font-size:0.68rem; color:#2a4060; text-align:center; line-height:1.8;">
        PowerTech Solutions © 2025<br>
        AURORA v1.0 — SENAI
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
agora = datetime.now().strftime("%d/%m/%Y | %H:%M:%S")

st.markdown(f"""
<div class="hero-container">
<div class="hero-left">
<div class="hero-logo">
❄️
</div>
<div>
<div class="hero-title">
AURORA AI
</div>
<div class="hero-subtitle">
ASSISTENTE TÉCNICO INTELIGENTE PARA
SISTEMAS DE REFRIGERAÇÃO INDUSTRIAL
</div>
<div class="hero-badges">
<span>⚡ NVIDIA AI</span>
<span>🔍 RAG ENGINE</span>
<span>📚 FAISS</span>
</div>
</div>
</div>
<div class="hero-status">
<div class="status-dot-big"></div>
<div>
<div class="status-title">
STATUS: ONLINE
</div>
<div class="status-time">
{agora}
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ── Validações ────────────────────────────────────────────────────────────────
if not api_ok:
    st.error("⚠️ **NVIDIA_API_KEY** não encontrada. Adicione ao `.env`.")
    st.stop()

if not pdf_ok:
    st.error("⚠️ `manual.pdf` não encontrado na pasta do projeto.")
    st.stop()

# ── RAG ───────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def inicializar_rag(api_key):
    loader = PyPDFLoader("manual.pdf")
    paginas = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = splitter.split_documents(paginas)
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=api_key,
        model_type="passage",
    )
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

try:
    with st.spinner("⚙️ Processando manual e inicializando vetores..."):
        retriever = inicializar_rag(nvidia_api_key)
    rag_ok = True
except Exception as e:
    st.error(f"❌ Erro ao inicializar RAG: `{e}`")
    rag_ok = False

# ── LLM + Chat ────────────────────────────────────────────────────────────────
if rag_ok:
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        nvidia_api_key=nvidia_api_key,
        temperature=0.2,
        max_tokens=1024,
    )

    template_prompt = """
PERSONA:
Você é a AURORA, uma engenheira especialista em manutenção e diagnóstico de sistemas de refrigeração industrial da PowerTech Solutions. Você é preciso, técnico e direto.

TAREFA:
Auxiliar técnicos de campo a diagnosticar alarmes, identificar causas de falhas, executar procedimentos de manutenção preventiva/corretiva e garantir a segurança operacional dos equipamentos.

CONTEXTO:
As informações abaixo foram extraídas do manual técnico oficial do equipamento e podem estar em inglês. Analise o conteúdo em inglês, mas SEMPRE responda em PORTUGUÊS DO BRASIL, de forma clara e estruturada.

RESTRIÇÕES:
- Use EXCLUSIVAMENTE as informações do manual fornecido no contexto.
- Se a informação não constar no manual, responda: "⚠️ Esta informação não consta no manual técnico. Recomendo contato com o suporte da PowerTech Solutions."
- Nunca invente dados técnicos, códigos de erro ou procedimentos.
- Para procedimentos de segurança, sempre incluir alertas de risco quando aplicável.

CONTEXTO DO MANUAL:
{context}

PERGUNTA DO TÉCNICO: {question}

RESPOSTA TÉCNICA (em português):
"""

    prompt_template = ChatPromptTemplate.from_template(template_prompt)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Olá! Sou a **AURORA**, sua assistente técnica especializada em sistemas de refrigeração industrial. ❄️\n\n"
                    "Posso ajudar com:\n"
                    "- 🔴 Significado de alarmes e códigos de erro\n"
                    "- 🔧 Procedimentos de manutenção preventiva e corretiva\n"
                    "- ⚡ Diagnóstico de falhas\n"
                    "- 🔒 Recomendações de segurança\n"
                    "- 🔄 Procedimentos de reinicialização\n\n"
                    "Como posso ajudar?"
                )
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_usuario := st.chat_input("Ex: O que significa o alarme E-04?"):
        st.session_state.messages.append({"role": "user", "content": prompt_usuario})
        with st.chat_message("user"):
            st.markdown(prompt_usuario)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Consultando manual técnico..."):
                try:
                    resposta = rag_chain.invoke(prompt_usuario)
                    st.markdown(resposta)
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                except Exception as e:
                    erro_msg = f"❌ Erro: `{e}`"
                    st.error(erro_msg)
                    st.session_state.messages.append({"role": "assistant", "content": erro_msg})