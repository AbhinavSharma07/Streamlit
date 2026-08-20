import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import tempfile
import os

# ----------------------------
# Helper functions
# ----------------------------

@st.cache_data(show_spinner=True)
def load_auto_documents(files):
    """Load automobile-related documents."""
    documents = []
    for file in files:
        ext = os.path.splitext(file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        documents.extend(loader.load())
    return documents

@st.cache_data(show_spinner=True)
def build_vectorstore(documents, embedding_model):
    """Split and embed documents, return FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    vectordb = FAISS.from_documents(docs, embedding_model)
    return vectordb

@st.cache_resource
def get_llm():
    return OpenAI(temperature=0)

def generate_prompt(question, context):
    """Craft a prompt specialized for automobile questions."""
    return f"""
You are an automotive expert AI assistant. Use the following context from automobile manuals, repair guides, or spec sheets to answer the user's question. Be concise and accurate.

Context:
{context}

User Question: {question}

Answer:
"""

# ----------------------------
# Streamlit App UI
# ----------------------------

def main():
    st.set_page_config(page_title="🚗 Automobile RAG App", layout="wide")
    st.title("🚗 Automobile Knowledge Assistant (RAG)")
    st.markdown("Upload automobile manuals, spec sheets, or guides and ask questions about cars!")

    uploaded_files = st.file_uploader("Upload automobile documents (PDF or TXT)", accept_multiple_files=True)

    if uploaded_files:
        docs = load_auto_documents(uploaded_files)
        embeddings = OpenAIEmbeddings()
        vectordb = build_vectorstore(docs, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        llm = get_llm()

        st.divider()
        st.header("🔍 Ask a question about the documents")
        user_question = st.text_input("Type your automobile-related question here:")

        if user_question:
            with st.spinner("Analyzing documents..."):
                # Retrieve context
                relevant_docs = retriever.get_relevant_documents(user_question)
                context_text = "\n\n".join([doc.page_content[:500] for doc in relevant_docs])
                prompt = generate_prompt(user_question, context_text)
                answer = llm(prompt)

            st.subheader("Answer:")
            st.write(answer)

            with st.expander("🔎 View Retrieved Document Snippets"):
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"**Document {i+1}:**")
                    st.write(doc.page_content[:400] + " ...")

    else:
        st.info("Upload automobile-related documents to get started.")

if __name__ == "__main__":
    main()
