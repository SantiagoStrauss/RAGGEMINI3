import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from typing import List, Tuple

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="RAG por Santiago Ramos", layout="wide")
st.title("RAG por Santiago Ramos")

# List of PDF files to load
PDF_FILES = ["NICSP en su bolsillo 2020.pdf"]

# Constants for models and settings
EMBEDDING_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class BM25Retriever:
    def __init__(self, documents: List[str]):
        self.tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_docs = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)[:top_k]
        return top_docs

@st.cache_data
def load_and_split_pdfs(pdf_files):
    """Load PDF files and split them into manageable chunks."""
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for pdf_file in pdf_files:
        try:
            with st.spinner(f"Loading PDF file: {pdf_file}"):
                loader = PyPDFLoader(pdf_file)
                data = loader.load()
                docs = text_splitter.split_documents(data)
                all_docs.extend(docs)
            st.success(f"Successfully loaded {pdf_file}")
        except Exception as e:
            st.error(f"Error loading {pdf_file}: {e}")
    return all_docs

@st.cache_resource
def create_retrieval_systems(_docs):
    """Create a vector store and BM25 Retriever from the provided documents."""
    try:
        with st.spinner("Creating retrieval systems..."):
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = Chroma.from_documents(documents=_docs, embedding=embeddings)
            
            # Create BM25 Retriever
            doc_texts = [doc.page_content for doc in _docs]
            bm25_retriever = BM25Retriever(doc_texts)
            
        st.success("Retrieval systems created successfully")
        return vectorstore, bm25_retriever, _docs
    except Exception as e:
        st.error(f"Error creating retrieval systems: {e}")
        return None, None, None

def get_relevant_context(vectorstore, bm25_retriever, docs, query, chat_history):
    """Retrieve relevant context using both vectorstore and BM25."""
    combined_query = query + " " + " ".join([m["content"] for m in chat_history])
    
    # Vector-based retrieval
    vector_results = vectorstore.similarity_search(combined_query, k=3)
    vector_docs = set([doc.page_content for doc in vector_results])
    
    # BM25 retrieval
    bm25_results = bm25_retriever.retrieve(combined_query, top_k=3)
    bm25_docs = set([docs[idx].page_content for idx, _ in bm25_results])
    
    # Combine results
    combined_docs = list(vector_docs.union(bm25_docs))
    
    return "\n".join(combined_docs)

def main():
    # Load and split documents
    all_docs = load_and_split_pdfs(PDF_FILES)

    # Create the retrieval systems
    vectorstore, bm25_retriever, docs = create_retrieval_systems(all_docs)
    if not vectorstore or not bm25_retriever:
        return

    # Initialize the language model
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0, max_tokens=None, timeout=None)

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context and chat history to answer the question."),
        ("human", "Context: {context}"),
        ("human", "Chat history: {chat_history}"),
        ("human", "Question: {question}")
    ])

    # Create LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Streamlit chat interface
    st.subheader("Ask a question about the loaded documents")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if query := st.chat_input("Your question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                with st.spinner("Thinking..."):
                    context = get_relevant_context(vectorstore, bm25_retriever, docs, query, st.session_state.messages)
                    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    response = llm_chain.run(context=context, chat_history=chat_history, question=query)
                    full_response = response
                    message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error processing query: {e}")
                full_response = f"I'm sorry, but I encountered an error: {str(e)}"
                message_placeholder.markdown(full_response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()