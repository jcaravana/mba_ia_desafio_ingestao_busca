"""
Script de busca (RAG).

Recebe a pergunta do usuário, vetoriza essa pergunta, busca no banco vetorial
PGVector os trechos do PDF mais parecidos com ela e usa o LLM para responder
com base somente nesse contexto recuperado.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Carrega as variáveis de ambiente definidas no arquivo .env (chaves de API, URL do banco, etc.)
load_dotenv()

# Template do prompt enviado ao LLM. {resultado_concatenado_bd} é o contexto
# recuperado do banco vetorial e {pergunta_usuario} é a pergunta do usuário.
# As regras deixam explícito que o modelo só pode responder com base no CONTEXTO.
query = """CONTEXTO:
    {resultado_concatenado_bd}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
    {pergunta_usuario}

RESPONDA A "PERGUNTA DO USUÁRIO" """

# Modelo de embeddings usado para vetorizar a pergunta do usuário
# (precisa ser o mesmo modelo usado na ingestão, para os vetores serem comparáveis).
# Deixe apenas uma das opções abaixo descomentada (a mesma usada em ingestion.py).
#embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL", "text-embedding-3-small"))
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview"))

# Conexão com o banco vetorial (PGVector) onde os embeddings foram gravados na ingestão
store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

# Modelo de chat (LLM) usado para gerar a resposta final a partir do contexto encontrado
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini"), temperature=0)


def consultar_dados(pergunta_do_usuario):
    # Busca no banco vetorial o contexto relevante para a pergunta e usa o LLM
    # para responder ao usuário somente com base nesse contexto.

    # Vetoriza a pergunta do usuário (mesmo processo usado para os chunks na ingestão)
    vetor_pergunta = embeddings.embed_query(pergunta_do_usuario)

    # Busca no banco vetorial os k chunks mais similares ao vetor da pergunta,
    # retornando cada chunk junto com sua pontuação de similaridade
    resultados = store.similarity_search_with_score_by_vector(vetor_pergunta, k=10)

    # Concatena o texto dos chunks encontrados em um único bloco de contexto
    resultado_concatenado_bd = "\n\n".join(
        doc.page_content for doc, _score in resultados
    )

    # Preenche o template do prompt com o contexto encontrado e a pergunta do usuário
    prompt = query.format(
        resultado_concatenado_bd=resultado_concatenado_bd,
        pergunta_usuario=pergunta_do_usuario,
    )

    # Envia o prompt para o LLM e retorna apenas o texto da resposta para ser consumido pelo chat do usuário
    resposta = llm.invoke(prompt)
    return resposta.content


if __name__ == "__main__":
    pergunta = input("Digite sua pergunta: ")
    print(consultar_dados(pergunta))
