# -*- coding: utf-8 -*-

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# ========== Settings ==========
DB_DIR = "chroma_db"
PDF_PATH = "data/HSC26-Bangla1st-Paper.pdf"
EMBED_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"
LLM_MODEL = "google/flan-t5-base"  # Or use: "google/mt5-small" for more multilingual support

# ========== Step 1: Load and Split PDF ==========
def load_documents():
    loader = PyPDFLoader(PDF_PATH)
    # loader = PyPDFLoader(PDF_PATH, encoding="utf-8")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    # # print some examples of the raw documents
    # print(f"[INFO] Loaded {len(raw_docs)} raw documents from PDF.")
    # for i, doc in enumerate(raw_docs):  # Show first 3 documents
    #     print(f"[INFO] Raw Document {i+1}: {doc.page_content[:100]}...")
    return splitter.split_documents(raw_docs)

# ========== Step 2: Build Vectorstore with Chroma ==========
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=DB_DIR)
    vectorstore.persist()
    return vectorstore

# ========== Step 3: Load LLM ==========
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return HuggingFacePipeline(pipeline=pipe)

# ========== Step 4: Setup RAG ==========
def setup_qa(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain

# ========== Step 5: Main ==========
def main():
    if not os.path.exists(DB_DIR):
        print("[INFO] Building vector DB from PDF...")
        documents = load_documents()
        vectorstore = build_vectorstore(documents)
    else:
        print("[INFO] Loading existing vector DB...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    llm = load_llm()
    qa = setup_qa(vectorstore, llm)

    while True:
        query = input("\n🧠 Enter your question (Bangla or English): ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa(query)
        print("\n🔍 Answer:\n", result['result'])

if __name__ == "__main__":
    main()
