"""
Chat via linha de comando (CLI).

Permite que o usuário faça perguntas repetidamente pelo terminal. Cada
pergunta é enviada para consultar_dados (src/search.py), que busca o
contexto no banco vetorial e retorna a resposta gerada pelo LLM.
"""

from search import consultar_dados

SAIR = {"sair", "exit", "quit"}


def main():
    print('Digite sua pergunta ("sair" para encerrar).')

    while True:
        pergunta = input("PERGUNTA: ").strip()

        if pergunta.lower() in SAIR:
            break

        if not pergunta:
            continue

        resposta = consultar_dados(pergunta)
        print(f"RESPOSTA: {resposta}")


if __name__ == "__main__":
    main()
