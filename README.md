# Ingestão e Busca Semântica com LangChain e Postgres

Desafio técnico do MBA em Engenharia de Software com IA. O projeto entrega um software capaz de:

- **Ingestão**: ler um arquivo PDF e salvar suas informações em um banco de dados PostgreSQL com a extensão `pgvector`.
- **Busca**: permitir que o usuário faça perguntas via linha de comando (CLI) e receba respostas baseadas apenas no conteúdo do PDF.

## Como funciona

1. `src/ingestion.py` lê o PDF `SuperTechIABrazil_Relatorio_2025.pdf`, divide o texto em chunks, gera o embedding de cada chunk e grava tudo no banco vetorial PGVector.
2. `src/search.py` recebe uma pergunta, vetoriza essa pergunta, busca no PGVector os chunks mais similares e usa um LLM para responder **somente** com base nesse contexto recuperado (RAG). Se a informação não estiver no PDF, o modelo responde que não tem informações suficientes.
3. `src/chat.py` é a interface de linha de comando: fica em loop pedindo perguntas ao usuário e exibindo as respostas geradas por `search.py`.

## Estrutura do projeto

```
.
├── docker-compose.yml                     # Postgres + extensão pgvector
├── requirements.txt                       # Dependências Python
├── .env.example                           # Modelo de variáveis de ambiente
├── SuperTechIABrazil_Relatorio_2025.pdf   # PDF usado na ingestão
└── src/
    ├── ingestion.py                       # Lê o PDF e grava os embeddings no PGVector
    ├── search.py                          # Busca por similaridade + resposta do LLM
    └── chat.py                            # CLI de perguntas e respostas
```

## Pré-requisitos

- [Docker](https://www.docker.com/) e Docker Compose
- Python 3.10+
- Uma chave de API da OpenAI (usada para embeddings e para o chat)

## Configuração

1. Crie e ative um ambiente virtual:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Copie o `.env.example` para `.env` e preencha as variáveis:

   ```bash
   cp .env.example .env
   ```

   | Variável | Descrição |
   |---|---|
   | `OPENAI_API_KEY` | Chave de API da OpenAI |
   | `OPENAI_MODEL` | Modelo de embeddings (ex.: `text-embedding-3-small`) |
   | `PGVECTOR_URL` | String de conexão do Postgres (ex.: `postgresql+psycopg://postgres:postgres@localhost:5432/rag`) |
   | `PGVECTOR_COLLECTION` | Nome da coleção usada no PGVector |

## Ordem de execução

1. Subir o banco de dados:

   ```bash
   docker compose up -d
   ```

2. Executar a ingestão do PDF:

   ```bash
   python src/ingestion.py
   ```

3. Rodar o chat:

   ```bash
   python src/chat.py
   ```

   Digite suas perguntas no prompt `PERGUNTA:`. Para encerrar, digite `sair`, `exit` ou `quit`.

## Stack

- [LangChain](https://www.langchain.com/) / [LangChain Community](https://pypi.org/project/langchain-community/) — carregamento e divisão do PDF em chunks
- [langchain-postgres](https://pypi.org/project/langchain-postgres/) + [pgvector](https://github.com/pgvector/pgvector) — armazenamento e busca vetorial no Postgres
- [langchain-openai](https://pypi.org/project/langchain-openai/) — embeddings e LLM (OpenAI)
- [Docker Compose](https://docs.docker.com/compose/) — Postgres com a extensão `pgvector` já habilitada
