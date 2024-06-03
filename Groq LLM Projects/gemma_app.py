import os
import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key,
               model='Gemma-7b-it')

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        st.session_state.loader = PyPDFDirectoryLoader(r"C:\Users\ADMIN\Desktop\LANGCHAIN\groq\us_census") ## Data ingestion
        st.session_state.docs = st.session_state.loader.load() ## document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                                        chunk_overlap=200) ## chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) ## splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
                                                        st.session_state.embeddings) ## vector store using openai embeddings


prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions: {input}
"""
)

prompt1 = st.text_input("Enter your question from the documents")

if st.button("Document Embeddings"):
    vector_embeddings()
    st.write("Vector Store db is ready!")


## create a prompt
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrievel_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrievel_chain.invoke({"input":prompt1})
    print("Response time :", time.process_time()-start)
    st.write(response['answer'])

    #with a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevent chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------------------")
