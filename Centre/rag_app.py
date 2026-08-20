import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import tempfile

# --- Helper / Setup functions ---

@st.cache_data(show_spinner=True)
def load_documents(files):
    """Load documents from uploaded files into LangChain Documents."""
    docs = []
    for file in files:
        fname = file.name.lower()
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tf:
            tf.write(file.getvalue())
            temp_path = tf.name

        # Choose loader based on extension
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")
        docs.extend(loader.load())
    return docs

@st.cache_data(show_spinner=True)
def build_vectorstore(docs, embedding_model):
    """Split docs, embed them, and build a vector store."""
    # split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    # embed and index
    vectorstore = FAISS.from_documents(splits, embedding_model)
    return vectorstore

@st.cache_resource(show_spinner=True)
def get_llm():
    """Return an LLM instance (OpenAI here)."""
    return OpenAI(temperature=0)

def main():
    st.set_page_config(page_title="RAG Chat App", layout="wide")
    st.title("🔍 RAG-powered Document Chat")

    st.markdown(
        """
        Upload documents, then ask questions about their content.  
        The app will retrieve relevant context and use an LLM to answer.
        """
    )

    # Sidebar: credentials & upload
    with st.sidebar:
        st.header("Upload & Config")
        files = st.file_uploader("Upload PDF or text files", accept_multiple_files=True)
        # Optional: let user choose embedding / LLM options
        # e.g. model names, vector store type
        st.info("Upload docs then switch to Chat tab")

    # If documents uploaded, process
    if files:
        docs = load_documents(files)
        embedding_model = OpenAIEmbeddings()  # or your embeddings
        vectorstore = build_vectorstore(docs, embedding_model)

        # Prepare QA chain
        llm = get_llm()
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        )

        # Chat / QA interface
        st.header("💬 Ask a question")
        question = st.text_input("Your question about the documents:")
        if question:
            with st.spinner("Thinking..."):
                answer = qa.run(question)
            st.subheader("Answer")
            st.write(answer)

            # Optionally show which docs were used (context)
            with st.expander("Show Retrieved Context"):
                docs_used = qa.retriever.get_relevant_documents(question)
                for i, d in enumerate(docs_used):
                    st.markdown(f"**Source {i + 1}:**")
                    st.write(d.page_content)

    else:
        st.info("Please upload one or more documents to get started.")

if __name__ == "__main__":
    main()
