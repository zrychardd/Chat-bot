import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# ── Layout ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistente de Manuais PDF",
    page_icon="📘",
    layout="centered",
)

st.markdown("""
<style>
    /* Fundo geral */
    .stApp { background-color: #0f1117; }

    /* Header customizado */
    .header-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #76b900;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .header-box h1 {
        color: #76b900;
        font-size: 1.6rem;
        margin: 0 0 4px 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .header-box p {
        color: #8a9ab5;
        margin: 0;
        font-size: 0.9rem;
    }

    /* Mensagens do chat */
    .stChatMessage {
        background-color: #1a1f2e !important;
        border-radius: 10px !important;
        border: 1px solid #2a3045 !important;
        margin-bottom: 8px !important;
    }

    /* Input */
    .stChatInputContainer {
        border-top: 1px solid #2a3045 !important;
        padding-top: 12px !important;
    }

    /* Badge verde NVIDIA */
    .badge {
        display: inline-block;
        background: #76b900;
        color: #0f1117;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        margin-top: 6px;
    }

    /* Spinner / info */
    .stSpinner > div { color: #76b900 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>📘 Assistente de Manuais PDF</h1>
    <p>Faça perguntas sobre o manual técnico — respondo sempre em português.</p>
    <span class="badge">⚡ NVIDIA AI</span>
</div>
""", unsafe_allow_html=True)

# ── API Key ──────────────────────────────────────────────────────────────────
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")

try:
    if "NVIDIA_API_KEY" in st.secrets:
        nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
except Exception:
    pass

# FIX: lógica estava invertida — parava quando tinha chave
if not nvidia_api_key:
    st.error("⚠️ Chave NVIDIA_API_KEY não encontrada. Adicione ao arquivo .env ou ao Secrets.")
    st.stop()

# ── RAG ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Processando o PDF, aguarde...")
def inicializar_rag():
    nome_arquivo = "manual.pdf"

    # FIX: verificava a string literal "nome_arquivo" em vez da variável
    if not os.path.exists(nome_arquivo):
        st.error(f"Arquivo '{nome_arquivo}' não encontrado na pasta do projeto.")
        st.stop()  # FIX: faltava ()

    loader = PyPDFLoader(nome_arquivo)
    paginas = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,  # FIX: era chunk_olverlap
    )

    docs = text_splitter.split_documents(paginas)

    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=nvidia_api_key,
        model_type="passage",
    )

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})  # FIX: "K" → "k"

retriever = inicializar_rag()

llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    nvidia_api_key=nvidia_api_key,
    temperature=0.2,
)

template_prompt = """
Você é um assistente técnico especializado e prestativo.
Os textos de contexto abaixo foram extraídos de um manual de produtos/serviços e podem estar em INGLÊS.
Analise o contexto mesmo que esteja em inglês, mas responda OBRIGATORIAMENTE EM PORTUGUÊS DO BRASIL.

Use estritamente as informações fornecidas. Se a resposta não constar no manual, diga:
"Desculpe, mas essa informação não consta no manual."

Contexto:
{context}

Pergunta: {question}
Resposta em português:
"""
# FIX: "question" estava fora das chaves no template original

prompt = ChatPromptTemplate.from_template(template_prompt)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Manual carregado com sucesso. O que você deseja saber? 😊"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt_usuario := st.chat_input("Ex: Qual o significado do código de erro 4?"):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.write(prompt_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Consultando o manual técnico..."):
            try:
                resposta = rag_chain.invoke(prompt_usuario)
                st.write(resposta)
                # FIX: era .messages(...) com () em vez de .append(...)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                st.error(f"Erro ao processar a requisição da API: {e}")