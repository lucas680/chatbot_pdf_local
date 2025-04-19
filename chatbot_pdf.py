import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Caminho da pasta com PDFs
pasta_pdfs = "./pdfs"
pdf_paths = [os.path.join(pasta_pdfs, nome) for nome in os.listdir(pasta_pdfs) if nome.endswith(".pdf")]

# Carregar documentos
all_documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs = loader.load()
    all_documents.extend(docs)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(all_documents, embedding_model)
retriever = db.as_retriever()

# Modelo leve
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

# Criar o chatbot
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Loop de conversa
print("🤖 Chatbot iniciado! Faça perguntas sobre os PDFs. Digite 'sair' para encerrar.\n")
while True:
    pergunta = input("Você: ")
    if pergunta.lower() == "sair":
        print("🤖 Até logo, senhor Lucas.")
        break
    resposta = qa.run(f"{pergunta}")
    print("🤖 IA:", resposta, "\n")
