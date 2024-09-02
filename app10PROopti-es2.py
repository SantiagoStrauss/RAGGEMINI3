pip install https://huggingface.co/spacy/es_core_news_sm/resolve/main/es_core_news_sm-any-py3-none-any.whl
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from typing import List, Tuple, Optional
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import numpy as np
import json
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from langchain.text_splitter import SpacyTextSplitter
from collections import defaultdict
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import voyageai
from langchain.embeddings.base import Embeddings

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Cargar variables de entorno y configurar el registro
load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuración de la aplicación Streamlit
st.set_page_config(page_title="Sistema de Recuperación de Documentos", layout="wide")
st.title("Sistema de Recuperación de Documentos")

# Configuración
PDF_FILES = ["NICSP en su bolsillo 2020.pdf", "Resolución 533 de 2015 Contaduría General de la Nación.pdf", "resolucion_contaduria_0414_2014.pdf"]
EMBEDDING_MODEL = "voyage-multilingual-2"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-1.5-flash")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Cargar modelo de lenguaje español
nlp = spacy.load("es_core_news_sm")

class VoyageAIEmbeddings(Embeddings):
    def __init__(self, model: str = "voyage-multilingual-2", batch_size: int = 128):
        self.model = model
        self.client = voyageai.Client()
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            results = self.client.embed(batch, model=self.model, input_type="document")
            all_embeddings.extend(results.embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.embed([text], model=self.model, input_type="query")
        return result.embeddings[0]

class BM25L:
    def __init__(self, corpus, k1=1.5, b=0.75, delta=0.5):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.avg_doc_len = sum(len(doc) for doc in corpus) / len(corpus)
        self.doc_freqs = []
        self.idf = defaultdict(float)
        self.doc_len = []
        self.corpus_size = len(corpus)
        self._initialize()

    def _initialize(self):
        for document in self.corpus:
            self.doc_len.append(len(document))
            freq_dict = defaultdict(int)
            for token in document:
                freq_dict[token.text] += 1
            self.doc_freqs.append(freq_dict)
            for word in freq_dict:
                self.idf[word] += 1

        for word, freq in self.idf.items():
            self.idf[word] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query):
        scores = [0] * self.corpus_size
        q_freqs = defaultdict(int)
        for token in query:
            q_freqs[token.text] += 1

        for i, doc in enumerate(self.corpus):
            for token in query:
                term = token.text
                if term not in self.doc_freqs[i]:
                    continue
                freq = self.doc_freqs[i][term]
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_doc_len)
                scores[i] += (numerator / denominator) + self.delta

        return scores

class BM25LRetriever:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75, delta: float = 0.5):
        self.tokenized_docs = [nlp(doc) for doc in documents]
        self.bm25 = BM25L(self.tokenized_docs, k1=k1, b=b, delta=delta)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokenized_query = nlp(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        return heapq.nlargest(top_k, enumerate(doc_scores), key=lambda x: x[1])

def preprocess_spanish_text(text: str) -> str:
    doc = nlp(text)
    processed_text = " ".join([token.lemma_.lower() for token in doc 
                               if not token.is_punct and not token.is_space 
                               and token.text.lower() not in STOP_WORDS])
    processed_text = processed_text.replace('ia', 'inteligencia artificial')
    return processed_text

@st.cache_data
def load_pdf(pdf_file: str) -> List[Document]:
    try:
        st.text(f"Cargando archivo PDF: {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        data = loader.load()
        text_splitter = SpacyTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = text_splitter.split_documents(data)
        preprocessed_docs = [Document(page_content=preprocess_spanish_text(doc.page_content), metadata=doc.metadata) for doc in docs]
        logger.info("Cargado exitosamente %s", pdf_file)
        st.success(f"Cargado exitosamente {pdf_file}")
        return preprocessed_docs
    except Exception as e:
        logger.error("Error cargando %s: %s", pdf_file, str(e))
        st.error(f"Error cargando {pdf_file}: {str(e)}")
        return []

def compute_tfidf_weights(docs: List[str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(docs)
    return vectorizer

@st.cache_resource
def create_retrieval_systems(_docs: List[Document]) -> Tuple[Optional[Chroma], Optional[BM25LRetriever], Optional[List[Document]], Optional[TfidfVectorizer]]:
    try:
        with st.spinner("Creando sistemas de recuperación..."):
            voyage_embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL, batch_size=128)

            vectorstore = Chroma.from_documents(
                documents=_docs,
                embedding=voyage_embeddings
            )
            doc_texts = [doc.page_content for doc in _docs]
            bm25l_retriever = BM25LRetriever(doc_texts, k1=1.2, b=0.75, delta=0.5)
            tfidf_vectorizer = compute_tfidf_weights(doc_texts)
        logger.info("Sistemas de recuperación creados exitosamente")
        return vectorstore, bm25l_retriever, _docs, tfidf_vectorizer
    except Exception as e:
        logger.error("Error creando sistemas de recuperación: %s", str(e))
        st.error(f"Error creando sistemas de recuperación: {str(e)}")
        return None, None, None, None

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def prepare_query(query: str) -> str:
    return preprocess_spanish_text(query)

def weight_chat_history(chat_history: List[dict], decay_factor: float = 0.8) -> str:
    weighted_history = []
    for i, message in enumerate(reversed(chat_history[-5:])):
        weight = decay_factor ** i
        weighted_history.append(f"{weight:.2f} * {message['content']}")
    return " ".join(reversed(weighted_history))

def rerank_results(docs: List[str], query: str, cross_encoder: CrossEncoder, original_scores: List[float]) -> List[str]:
    pairs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(pairs)
    combined_scores = [0.7 * new_score + 0.3 * original_score for new_score, original_score in zip(scores, original_scores)]
    return [doc for _, doc in sorted(zip(combined_scores, docs), reverse=True)]

def fallback_keyword_search(docs: List[Document], query: str) -> str:
    keywords = query.lower().split()
    relevant_docs = []
    for doc in docs:
        if any(keyword in doc.page_content.lower() for keyword in keywords):
            relevant_docs.append(doc.page_content)
    if not relevant_docs:
        return "No pude encontrar información relevante. ¿Puedes reformular tu pregunta?"
    return "\n".join(relevant_docs[:3])

async def get_relevant_context(vectorstore: Chroma, bm25l_retriever: BM25LRetriever, docs: List[Document], tfidf_vectorizer: TfidfVectorizer, query: str, chat_history: List[dict], cross_encoder: CrossEncoder) -> str:
    try:
        weighted_history = weight_chat_history(chat_history)
        combined_query = query + " " + weighted_history
        prepared_query = prepare_query(combined_query)
        
        async def vector_search():
            try:
                voyage_embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL)
                query_embedding = voyage_embeddings.embed_query(prepared_query)
                return vectorstore.similarity_search_by_vector(query_embedding, k=10)
            except Exception as e:
                logger.error(f"Búsqueda vectorial fallida: {str(e)}")
                return []
        
        async def bm25l_search():
            try:
                return bm25l_retriever.retrieve(prepared_query, top_k=10)
            except Exception as e:
                logger.error(f"Búsqueda BM25L fallida: {str(e)}")
                return []
        
        vector_results, bm25l_results = await asyncio.gather(vector_search(), bm25l_search())
        
        if not vector_results and not bm25l_results:
            logger.warning("Ambas búsquedas, vectorial y BM25L, fallaron. Recurriendo a búsqueda por palabras clave.")
            return fallback_keyword_search(docs, prepared_query)
        
        combined_results = []
        for doc in vector_results:
            heapq.heappush(combined_results, (-0.3, doc.page_content))
        
        for idx, score in bm25l_results:
            doc_content = docs[idx].page_content
            heapq.heappush(combined_results, (-0.8 * score, doc_content))
        
        tfidf_scores = tfidf_vectorizer.transform([prepared_query]).toarray()[0]
        for idx, doc in enumerate(docs):
            doc_content = doc.page_content
            if any(doc_content == content for _, content in combined_results):
                heapq.heappush(combined_results, (-0.3 * tfidf_scores[idx], doc_content))
        
        top_results = heapq.nsmallest(10, combined_results)
        docs_to_rerank = [doc for _, doc in top_results]
        original_scores = [-score for score, _ in top_results]
        
        reranked_docs = rerank_results(docs_to_rerank, prepared_query, cross_encoder, original_scores)
        
        return "\n".join(reranked_docs[:5])
    
    except Exception as e:
        logger.error(f"Error en get_relevant_context: {str(e)}")
        return fallback_keyword_search(docs, query)

def similarity(text1: str, text2: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

async def iterative_retrieval(vectorstore: Chroma, bm25l_retriever: BM25LRetriever, docs: List[Document], tfidf_vectorizer: TfidfVectorizer, query: str, chat_history: List[dict], cross_encoder: CrossEncoder, llm: ChatGoogleGenerativeAI, max_iterations: int = 3) -> str:
    combined_context = ""
    current_query = query

    for _ in range(max_iterations):
        context = await get_relevant_context(vectorstore, bm25l_retriever, docs, tfidf_vectorizer, current_query, chat_history, cross_encoder)
        combined_context += "\n" + context

        follow_up_prompt = f"""Basado en el siguiente contexto y la consulta original, genera una pregunta de seguimiento para obtener más información relevante:
        
        Consulta original: {query}
        Contexto actual: {combined_context[:500]}
        
        Pregunta de seguimiento:"""
        
        follow_up_question = llm.predict(follow_up_prompt)
        
        if similarity(follow_up_question, query) > 0.8:
            break
        
        current_query = follow_up_question

    return combined_context

def save_feedback(query: str, answer: str, feedback: int):
    feedback_data = {
        "query": query,
        "answer": answer,
        "feedback": feedback
    }
    with open("feedback_log.json", "a") as f:
        json.dump(feedback_data, f)
        f.write("\n")

def main():
    all_docs = []
    for pdf_file in PDF_FILES:
        all_docs.extend(load_pdf(pdf_file))

    if not all_docs:
        st.error("No se cargaron documentos exitosamente. Por favor, verifica tus archivos PDF e inténtalo de nuevo.")
        return

    vectorstore, bm25l_retriever, docs, tfidf_vectorizer = create_retrieval_systems(all_docs)
    if not vectorstore or not bm25l_retriever or not tfidf_vectorizer:
        return

    streaming_handler = StreamingStdOutCallbackHandler()
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0, max_tokens=150, timeout=None, streaming=True, callbacks=[streaming_handler])

    cross_encoder = load_cross_encoder()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil. Usa el siguiente contexto y el historial de chat para responder la pregunta, prestando mucha atención a los detalles específicos de la pregunta y pensando paso a paso."),
        ("human", "Contexto: {context}"),
        ("human", "Historial de chat: {chat_history}"),
        ("human", "Pregunta: {question}")
    ])
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

    st.subheader("Haz una pregunta sobre los documentos")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Tu pregunta sobre los documentos:"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                with st.spinner("Analizando documentos..."):
                    context = asyncio.run(iterative_retrieval(vectorstore, bm25l_retriever, docs, tfidf_vectorizer, query, st.session_state.messages, cross_encoder, llm))
                    
                    response = llm_chain.predict(context=context, question=query)
                    full_response = response
                    message_placeholder.markdown(full_response)
            except Exception as e:
                logger.error("Error procesando la consulta: %s", str(e))
                full_response = f"Lo siento, pero encontré un error al procesar tu consulta. Esto es lo que encontré basado en una búsqueda por palabras clave: {fallback_keyword_search(docs, query)}"
                message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Mecanismo de retroalimentación
    if st.button("Enviar Retroalimentación"):
        feedback = st.slider("Califica la calidad de la respuesta (1-5)", 1, 5, 3)
        save_feedback(query, full_response, feedback)
        st.success("¡Gracias por tu retroalimentación!")

if __name__ == "__main__":
    main()
