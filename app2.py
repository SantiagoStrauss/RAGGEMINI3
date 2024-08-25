import streamlit as st
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.title("RAG Application by Santiago Ramos")

# Load multiple PDFs
pdf_files = ["yolov9_paper.pdf", "David Santiago Ramos CV en-output.pdf"]

# Initialize document list outside the loop for efficiency
all_docs = []

for pdf_file in pdf_files:
    try:
        st.write(f"Loading {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  # Add overlap to improve context
        docs = text_splitter.split_documents(data)
        all_docs.extend(docs)
        st.write(f"Loaded {len(docs)} documents from {pdf_file}")
    except Exception as e:
        st.error(f"Error loading {pdf_file}: {e}")

# Create the vectorstore
vectorstore = None
try:
    st.write("Creating embeddings...")
    embeddings = GooglePalmEmbeddings(model_name="models/text-multilingual-embedding-002") 
    st.write("Embeddings created successfully.")

    # Configure Chroma client settings
    chroma_settings = {
        "tenant": "valid_tenant",  # Ensure this tenant exists
        "database": "valid_database",  # Ensure this database exists
        "persist_directory": "./chroma_persist"  # Directory to persist the data
    }

    st.write("Creating vectorstore...")
    vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory=chroma_settings["persist_directory"])
    st.write("Vectorstore created successfully.")
except Exception as e:
    st.error(f"Error creating vectorstore: {e}")

# Initialize GooglePalm model correctly
try:
    st.write("Initializing LLM...")
    llm = GooglePalm(model_name="models/llm-gecko-001", temperature=0, max_output_tokens=100)  # Adjust max_output_tokens as needed
    st.write("LLM initialized successfully.")
except Exception as e:
    st.error(f"Error initializing LLM: {e}")

# Streamlit UI
query = st.text_input("Ask me something: ")  # User-friendly prompt

# Create RetrievalQA chain
if query and vectorstore:
    try:
        st.write("Creating RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Retrieve top 3 relevant chunks
            return_source_documents=True  # Optionally return source documents for transparency
        )
        st.write("RetrievalQA chain created successfully.")

        # Get the answer and source documents
        st.write("Processing query...")
        result = qa_chain({"query": query})
        st.write(result['result'])

        # Optionally display source documents
        st.subheader("Source Documents:")
        for doc in result['source_documents']:
            st.write(doc.metadata['source'])
            st.write(doc.page_content)
            st.write("---")
    except Exception as e:
        st.error(f"Error processing query: {e}")
elif query:
    st.error("Vectorstore is not defined. Unable to process the query.")