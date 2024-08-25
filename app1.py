import streamlit as st
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Streamlit app title
st.title("RAG Application por Santiago Ramos")

# List of PDF files to load
pdf_files = ["yolov9_paper.pdf", "David Santiago Ramos CV en-output.pdf"]

# Function to load and split documents from PDFs
def load_and_split_pdfs(pdf_files):
    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        all_docs.extend(docs)
    return all_docs

# Load and split documents
all_docs = load_and_split_pdfs(pdf_files)

# Create the vectorstore
vectorstore = Chroma.from_documents(
    documents=all_docs, 
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)

# Streamlit chat input
query = st.chat_input("Di algo: ")

# Define system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Process query if provided
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])