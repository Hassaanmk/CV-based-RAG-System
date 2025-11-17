

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import JinaEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
import streamlit as st

# Fix OpenMP conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dotenv import load_dotenv
load_dotenv()

JINA_API = os.getenv("JINA_API_KEY")
GOOGLE_API = os.getenv("GOOGLE_API_KEY")

# vector_store = FAISS.load_local("faiss_index")
# retriever = vector_store.as_retriever()

# Initialize embeddings
embeddings = JinaEmbeddings(
    jina_api_key=JINA_API,
    model_name="jina-embeddings-v2-base-en"
)

vector_store = FAISS.load_local(
    "faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API,
)



def query_rag(question, vectorstore=vector_store, model=llm, k=10):
    """Query function for RAG"""
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=k)
    compressor = JinaRerank(jina_api_key=JINA_API, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=vectorstore.as_retriever(k=10))
    reranked_docs = compression_retriever.invoke(question) # get_relevant_documents(question) does not work in new versions
    # Build context
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    
    # Create messages for LLM
    messages = [
        (
            "system",
            "You are a helpful assistant that answers questions based on the provided context. The context is a document from CV of some individuals."
            "If you cannot answer based on the context, say so clearly."
            "Give answers based on context only."
        ),
        (
            "human",
            f"""Context:
{context}

Question: {question}"""
        )
    ]
    
    # Generate answer using invoke
    ai_msg = model.invoke(messages)
    
    return ai_msg.content, reranked_docs

# Streamlit UI
st.set_page_config(page_title="RAG System for ByteCorp Assignment", page_icon="🔍")
st.title("RAG System for ByteCorp Assignment")

# Main interface
question = st.text_input("Ask a question:", placeholder="Enter your question here...")

if question:
    with st.spinner("Generating answer..."):
        answer, docs = query_rag(question, vector_store, k=10)
        
        st.subheader("Answer:")
        st.write(answer)
        
        with st.expander("📚 Sources"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Source {i}:**")
                st.text(doc.page_content[:300] + "...")
                st.divider()