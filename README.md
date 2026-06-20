# ❄️ AURORA AI - Assistente Técnico Inteligente

![Status do Projeto](https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green?style=for-the-badge)
![NVIDIA AI](https://img.shields.io/badge/NVIDIA_AI-Powered-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

> **AURORA AI** é uma aplicação desenvolvida como projeto educacional (SENAI) para a empresa fictícia *PowerTech Solutions*. Trata-se de um sistema de inteligência artificial voltado para o diagnóstico e manutenção de **Sistemas de Refrigeração Industrial**.

<div align="center">

🔗 **[Acessar o site](https://chatbotaurora.streamlit.app/)**

</div>

---




## 💡 Sobre o Projeto

O objetivo deste projeto é fornecer uma ferramenta de suporte rápido e preciso para técnicos de campo. Utilizando a arquitetura **RAG (Retrieval-Augmented Generation)**, a AURORA "lê" o manual técnico do equipamento em PDF e responde a dúvidas complexas baseando-se **estritamente** na documentação oficial.

A interface foi projetada com um design *Dark Mode / Industrial*, utilizando painéis de status e efeitos neon para simular o painel de controle de uma central de refrigeração.

### 🛠️ O que a AURORA pode fazer?
* 🔴 Explicar o significado de alarmes e códigos de erro.
* 🔧 Orientar sobre procedimentos de manutenção preventiva e corretiva.
* ⚡ Auxiliar no diagnóstico de falhas mecânicas ou elétricas.
* 🔒 Fornecer recomendações de segurança baseadas no manual.
* 🔄 Detalhar procedimentos de reinicialização dos sistemas.

---

## 🚀 Tecnologias Utilizadas

O projeto foi construído utilizando as seguintes bibliotecas e ferramentas:

* **[Streamlit](https://streamlit.io/):** Criação da interface de usuário (Frontend) e roteamento interativo.
* **[LangChain](https://python.langchain.com/):** Orquestração dos modelos de IA, prompts e pipelines de dados.
* **[FAISS](https://faiss.ai/):** Banco de dados vetorial em memória para busca de similaridade ultrarrápida.
* **[NVIDIA AI Endpoints](https://build.nvidia.com/):** * *Modelo de Linguagem (LLM):* `meta/llama-3.1-8b-instruct`
    * *Embeddings:* `nvidia/nv-embedqa-e5-v5`
* **PyPDFLoader:** Para extração de texto do manual técnico em PDF.
* **CSS Customizado:** Injeção de CSS nativo para estilização avançada (Glassmorphism, responsividade, etc).

---

## ⚙️ Arquitetura e Funcionamento (RAG)

1. **Ingestão de Dados:** O sistema carrega o arquivo `manual.pdf` local.
2. **Processamento:** O texto é fatiado em pequenos blocos (chunks) usando o `RecursiveCharacterTextSplitter`.
3. **Vetorização:** Os blocos de texto são transformados em vetores matemáticos usando o modelo de Embeddings da NVIDIA e armazenados no FAISS.
4. **Consulta (Retriever):** Quando o técnico faz uma pergunta, o sistema busca os 4 blocos de texto do PDF mais relevantes para aquela dúvida.
5. **Geração (LLM):** O Llama-3.1 recebe a pergunta do usuário junto com o contexto extraído do manual e gera uma resposta técnica, precisa e traduzida para o Português do Brasil.

---

## 💻 Como rodar o projeto localmente

Siga os passos abaixo para testar o sistema na sua máquina.

### 1. Clone o repositório
```bash
git clone https://github.com/SEU-USUARIO/aurora-ai.git
cd aurora-ai
