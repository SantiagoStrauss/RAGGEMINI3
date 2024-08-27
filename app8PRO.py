__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import logging
import asyncio
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Optional
from langchain.schema import Document
from sentence_transformers import CrossEncoder

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="RAG por Santiago Ramos", layout="wide")
st.title("RAG por Santiago Ramos")

# Configuration
PDF_FILES = ["NICSP en su bolsillo 2020.pdf"]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-1.5-flash")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

class BM25Retriever:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=k1, b=b)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_docs = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)[:top_k]
        return top_docs

def preprocess_text(text: str) -> str:
    # Normaliza las variaciones de NICSP
    text = re.sub(r'\bNICSP\s*(\d+)', r'NICSP\1', text, flags=re.IGNORECASE)
    # Otros preprocesamientos que puedas necesitar
    return text

@st.cache_data
def load_pdf(pdf_file: str) -> List[Document]:
    """Load a single PDF file and split it into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    try:
        st.text(f"Loading PDF file: {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        data = loader.load()
        docs = text_splitter.split_documents(data)
        preprocessed_docs = [Document(page_content=preprocess_text(doc.page_content), metadata=doc.metadata) for doc in docs]
        logger.info("Successfully loaded %s", pdf_file)
        st.success(f"Successfully loaded {pdf_file}")
        return preprocessed_docs
    except Exception as e:
        logger.error("Error loading %s: %s", pdf_file, str(e))
        st.error(f"Error loading {pdf_file}: {str(e)}")
        return []

@st.cache_resource
def create_retrieval_systems(_docs: List[Document]) -> Tuple[Optional[Chroma], Optional[BM25Retriever], Optional[List[Document]]]:
    """Create a vector store and BM25 Retriever from the provided documents."""
    try:
        with st.spinner("Creating retrieval systems..."):
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = Chroma.from_documents(documents=_docs, embedding=embeddings)
            
            doc_texts = [doc.page_content for doc in _docs]
            bm25_retriever = BM25Retriever(doc_texts, k1=1.2, b=0.75)
            
        logger.info("Retrieval systems created successfully")
        return vectorstore, bm25_retriever, _docs
    except Exception as e:
        logger.error("Error creating retrieval systems: %s", str(e))
        st.error(f"Error creating retrieval systems: {str(e)}")
        return None, None, None

def expand_query(query: str) -> str:
    # Añade variaciones relevantes a la consulta
    if "NICSP" in query:
        match = re.search(r'\bNICSP\s*(\d+)', query)
        if match:
            nicsp_num = match.group(1)
            query += f" OR 'Norma Internacional de Contabilidad del Sector Público {nicsp_num}'"
    return query

def keyword_search(docs: List[Document], query: str, top_k: int = 5) -> List[Document]:
    keywords = re.findall(r'\b\w+\b', query.lower())
    scores = []
    for doc in docs:
        score = sum(1 for keyword in keywords if keyword in doc.page_content.lower())
        scores.append((doc, score))
    return [doc for doc, score in sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]]

async def get_relevant_context(vectorstore: Chroma, bm25_retriever: BM25Retriever, docs: List[Document], query: str, chat_history: List[dict]) -> str:
    """Retrieve relevant context using both vectorstore and BM25 with improved context handling."""
    recent_history = chat_history[-3:]
    combined_query = query + " " + " ".join([m["content"] for m in recent_history])
    expanded_query = expand_query(combined_query)
    
    async def vector_search():
        return vectorstore.similarity_search(expanded_query, k=10)
    
    async def bm25_search():
        return bm25_retriever.retrieve(expanded_query, top_k=10)
    
    vector_results, bm25_results = await asyncio.gather(vector_search(), bm25_search())
    
    logger.debug(f"Vector search results: {[doc.page_content[:100] for doc in vector_results]}")
    logger.debug(f"BM25 search results: {[docs[idx].page_content[:100] for idx, _ in bm25_results]}")
    
    vector_docs = set([doc.page_content for doc in vector_results])
    bm25_docs = set([docs[idx].page_content for idx, _ in bm25_results])
    keyword_results = keyword_search(docs, expanded_query)
    keyword_docs = set([doc.page_content for doc in keyword_results])
    
    combined_docs = list(vector_docs.union(bm25_docs).union(keyword_docs))
    reranked_docs = rerank_results(combined_docs, query)
    
    return "\n".join(reranked_docs[:5])

def rerank_results(docs: List[str], query: str) -> List[str]:
    """Re-rank results using a cross-encoder model."""
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    pairs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(pairs)
    
    ranked_results = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return ranked_results

async def two_step_search(vectorstore: Chroma, bm25_retriever: BM25Retriever, docs: List[Document], query: str, chat_history: List[dict]) -> str:
    # Paso 1: Encuentra la NICSP relevante
    nicsp_query = re.search(r'\bNICSP\s*(\d+)', query)
    if nicsp_query:
        nicsp_num = nicsp_query.group(1)
        nicsp_docs = [doc for doc in docs if f"NICSP {nicsp_num}" in doc.page_content]
        
        # Paso 2: Busca información específica dentro de los documentos de esa NICSP
        relevant_docs = keyword_search(nicsp_docs, query, top_k=3)
        return "\n".join([doc.page_content for doc in relevant_docs])
    else:
        # Si no se menciona una NICSP específica, usa la búsqueda normal
        return await get_relevant_context(vectorstore, bm25_retriever, docs, query, chat_history)

def main():
    # Load and split documents
    all_docs = []
    for pdf_file in PDF_FILES:
        all_docs.extend(load_pdf(pdf_file))

    if not all_docs:
        st.error("No documents were successfully loaded. Please check your PDF files and try again.")
        return

    # Create the retrieval systems
    vectorstore, bm25_retriever, docs = create_retrieval_systems(all_docs)
    if not vectorstore or not bm25_retriever:
        return

    # Initialize the language model with streaming
    streaming_handler = StreamingStdOutCallbackHandler()
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2, max_tokens=150, timeout=None, streaming=True, callbacks=[streaming_handler])

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant specialized in NICSP (Normas Internacionales de Contabilidad del Sector Público). Use the following context and chat history to answer the question, paying close attention to the specific details in the question."),
        ("human", "Context: {context}"),
        ("human", "Chat history: {chat_history}"),
        ("human", "Question: {question}")
    ])

    # Create LLM chain with improved memory
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=500,
        input_key="question",
        memory_key="chat_history",
        return_messages=True
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True,
        output_key="answer"
    )

    # Streamlit chat interface
    st.subheader("Ask a question about NICSP")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Your question about NICSP:"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                with st.spinner("Analyzing NICSP documents..."):
                    context = asyncio.run(two_step_search(vectorstore, bm25_retriever, docs, query, st.session_state.messages))
                    
                    response = llm_chain.predict(context=context, question=query)
                    full_response = response
                    message_placeholder.markdown(full_response)
            except Exception as e:
                logger.error("Error processing NICSP query: %s", str(e))
                full_response = f"I apologize, but I encountered an error while processing your NICSP query: {str(e)}"
                message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    if st.checkbox("Show context used for answer"):
        st.write(context)

if __name__ == "__main__":
    main()
