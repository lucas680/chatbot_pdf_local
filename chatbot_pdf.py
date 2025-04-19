import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Caminho da pasta onde estão os PDFs
pasta_pdfs = "./pdfs"
pdf_paths = [os.path.join(pasta_pdfs, nome) for nome in os.listdir(pasta_pdfs) if nome.endswith(".pdf")]

# Carrega todos os documentos dos PDFs
all_documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs = loader.load()
    all_documents.extend(docs)

# Cria os embeddings para os documentos
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(all_documents, embedding_model)

# Limita o número de trechos retornados para evitar excesso de tokens
retriever = db.as_retriever(search_kwargs={"k": 2})

# Modelo de linguagem leve (flan-t5-base)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Pipeline de geração de texto
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

# Criação do chatbot com QA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Loop de perguntas e respostas
print("🤖 Chatbot iniciado! Faça perguntas sobre os PDFs. Digite 'sair' para encerrar.\n")

while True:
    pergunta = input("Você: ")
    if pergunta.lower() == "sair":
        print("🤖 Até logo, senhor Lucas.")
        break

    # Início da contagem de tempo e exibição de "aguarde..."
    print("⏳ Aguarde...", end="\r")
    inicio = time.time()

    # Adiciona a instrução para responder em português
    resposta = qa.invoke(pergunta)

    fim = time.time()
    tempo = round(fim - inicio, 2)

    # Sobrescreve o print anterior e mostra a resposta com tempo
    print(" " * 60, end="\r")  # Limpa linha anterior
    print(f"🤖 IA: {resposta['result']} ({tempo}s)\n")
