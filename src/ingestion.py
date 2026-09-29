"""
Script de ingestão de dados.

Lê o PDF "SuperTechIABrazil_Relatorio_2025.pdf" localizado na raiz do projeto,
divide o texto em chunks menores, gera o embedding de cada chunk e grava
esses embeddings (junto com o texto e o metadata) no banco vetorial PGVector,
deixando os dados prontos para busca por similaridade.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

# Carrega as variáveis de ambiente definidas no arquivo .env (chaves de API, URL do banco, etc.)
print("[1/8] Carregando variáveis de ambiente...")
load_dotenv()

# Localiza o PDF na raiz do projeto (um nível acima da pasta src)
print("[2/8] Localizando o PDF...")
current_dir = Path(__file__).parent
pdf_path = current_dir.parent / "SuperTechIABrazil_Relatorio_2025.pdf"

# Lê o PDF e extrai o texto de cada página como um Document do LangChain
print(f"[3/8] Lendo o PDF ({pdf_path.name})...")
docs = PyPDFLoader(str(pdf_path)).load()
print(f"      {len(docs)} página(s) carregada(s).")

# Divide os documentos em pedaços (chunks) menores, com sobreposição entre eles,
# para facilitar a busca por similaridade e caber no contexto do modelo
print("[4/8] Dividindo o texto em chunks...")
splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False).split_documents(docs)
print(f"      {len(splits)} chunk(s) gerado(s).")

# Se o PDF não gerou nenhum chunk, não há o que indexar
if not splits:
    print("      Nenhum chunk gerado, encerrando ingestão.")
    raise SystemExit(0)

# Recria os documentos removendo do metadata os valores vazios ou nulos
print("[5/8] Limpando metadata dos chunks...")
enriched = [
    Document(
        page_content=d.page_content,
        metadata={
           k: v for k, v in d.metadata.items() if v not in ("", None)
        },
    )
    for d in splits
]

# Gera um id único e sequencial para cada chunk (usado como chave no banco vetorial)
print("[6/8] Gerando ids dos chunks...")
ids = [f"doc-{i}" for i in range(len(enriched))]

# Modelo de embeddings usado para transformar cada chunk de texto em um vetor numérico.
# Deixe apenas uma das opções abaixo descomentada (a mesma tem que ser usada na busca, em search.py).
print("[7/8] Conectando ao modelo de embeddings e ao banco vetorial...")
#embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL", "text-embedding-3-small"))
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview"))

# Conexão com o banco vetorial (PGVector) onde os embeddings serão armazenados.
# pre_delete_collection=True faz o truncate da coleção (apaga apenas os dados
# da coleção PGVECTOR_COLLECTION, sem afetar outras coleções do banco) antes
# de recriá-la, evitando duplicar dados a cada nova execução da ingestão.
print("      Truncando (limpando) a coleção antes de recriar...")
store = PGVector(
    embeddings = embeddings,
    collection_name = os.getenv("PGVECTOR_COLLECTION"),
    connection = os.getenv("PGVECTOR_URL"),
    use_jsonb = True,
    pre_delete_collection = True,
)

# Gera os embeddings de cada chunk e os grava no banco vetorial
print(f"[8/8] Gerando embeddings e gravando {len(enriched)} chunk(s) no banco vetorial...")
store.add_documents(documents=enriched, ids=ids)
print("      Ingestão concluída com sucesso!!")
