import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import tempfile
import shutil

app = FastAPI()

# Estrutura para armazenar o QA
qa = None

# Modelos
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

# Modelos de requisições
class Pergunta(BaseModel):
    pergunta: str

@app.post("/listPdfs")
async def list_pdfs(files: List[UploadFile] = File(...)):
    global qa

    # Cria pasta temporária para salvar os PDFs
    temp_dir = tempfile.mkdtemp()
    all_documents = []

    try:
        for pdf_file in files:
            temp_path = os.path.join(temp_dir, pdf_file.filename)
            with open(temp_path, "wb") as f:
                content = await pdf_file.read()
                f.write(content)

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            all_documents.extend(docs)

        # Cria nova base FAISS com embeddings
        db = FAISS.from_documents(all_documents, embedding_model)
        retriever = db.as_retriever(search_kwargs={"k": 2})

        # Atualiza o QA global
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

        return {"status": "PDFs carregados com sucesso!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        shutil.rmtree(temp_dir)

@app.post("/question")
async def question(pergunta: Pergunta):
    global qa
    if qa is None:
        return JSONResponse(status_code=400, content={"error": "Nenhum PDF carregado. Use /listPdfs primeiro."})

    resposta = qa.invoke(pergunta.pergunta)
    return {"resposta": resposta['result']}
