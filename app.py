import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']


st.title(" Fine-Tuned RAG Chatbot with Streaming Responses using Llama3")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


# ─── Sidebar: session info ──────────────────────────────────────────
st.sidebar.header("⚙️ Session Info")
st.sidebar.write("*Model in use:*", llm.model_name)

if "vectors" in st.session_state:
    # try to get FAISS internal count, fallback to docstore length
    try:
        n = st.session_state.vectors._faiss_index.ntotal
    except:
        # for LangChain community FAISS
        n = len(st.session_state.vectors.docstore._dict)
    st.sidebar.write("*Indexed docs:*", n)
# ────────────────────────────────────────────────────────────────────

# Vector embedding

def vector_embedding():

    #if "vectors" not in st.session_state:

    loader = PyPDFLoader("AI Training Document.pdf")
    docs   = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50,separators=[". ","? ","! ","\n\n","\n"," ",""],)
    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("No document chunks to embed — check your loader & splitter settings")
        return

    # now safe to embed
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    st.session_state.vectors = FAISS.from_documents(chunks, embedding_model)
    st.success("Vector store DB is ready")
    

prompt1 = st.text_input("Enter your Question from Documents")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector store DB is ready")


if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

# ─── Reset everything & rerun ───────────────────────────────────────
if st.sidebar.button("🔄 Reset Chat"):
    for key in list(st.session_state.keys()):
        if key == "vectors":
            continue
        del st.session_state[key]
    st.rerun()
# ────────────────────────────────────────────────────────────────────