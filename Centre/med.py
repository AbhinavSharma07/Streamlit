import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import tempfile
import os

# --- Utility functions ---

@st.cache_data(show_spinner=True)
def load_medical_docs(files):
    """
    Load uploaded medical documents (PDF or text) into LangChain documents.
    """
    docs = []
    for file in files:
        fname = file.name.lower()
        # Write to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tf:
            tf.write(file.getvalue())
            temp_path = tf.name
        # Choose loader
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")
        docs.extend(loader.load())
    return docs

@st.cache_data(show_spinner=True)
def build_medical_vectorstore(docs, embedding_model):
    """
    Split medical docs and create a vector store.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    return vectorstore

@st.cache_resource(show_spinner=True)
def get_medical_llm():
    """
    Returns a medical‑suitable LLM instance.
    """
    # You might choose a specialized medical model; here using OpenAI default
    return OpenAI(temperature=0)

def make_medical_prompt(question, context_docs):
    """
    Create a prompt template that emphasizes evidence, citations, and disclaimers.
    """
    template = """
You are a medical assistant. Use the provided context, which comes from medical literature or documents, to answer the user's question. Cite brief excerpts or page info when relevant.

Context:
{context}

User Question: {question}

Answer in a medically responsible way. If uncertain, say you are uncertain and recommend consulting a qualified health professional.
"""
    # join contexts
    ctx = "\n\n".join([doc.page_content for doc in context_docs])
    return template.format(context=ctx, question=question)

def run_medical_qa(qa_chain, question):
    """
    Run QA and return answer + doc citations.
    """
    # Get relevant docs
    docs = qa_chain.retriever.get_relevant_documents(question)
    prompt = make_medical_prompt(question, docs)
    # run LLM on prompt manually using chain LLM
    answer = qa_chain.llm(prompt)
    return answer, docs

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Medical RAG App", layout="wide")
    st.title("🩺 Medical Knowledge Q&A (RAG)")

    st.markdown("""
    Upload medical research papers, guidelines, or case studies (PDF / text).  
    Then ask medical questions based on them.  
    **Note**: This is for educational / reference use only and not a substitute for professional medical advice.
    """)

    uploaded = st.file_uploader("Upload medical documents (PDF or TXT)", accept_multiple_files=True)

    if uploaded:
        docs = load_medical_docs(uploaded)
        embedding_model = OpenAIEmbeddings()
        vecstore = build_medical_vectorstore(docs, embedding_model)

        llm = get_medical_llm()
        # Use a dummy RetrievalQA but we will override prompt
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vecstore.as_retriever(search_kwargs={"k": 5}),
        )

        st.header("Ask a medical question")
        question = st.text_input("Enter your question:")

        if question:
            with st.spinner("Retrieving and answering..."):
                answer, docs_used = run_medical_qa(qa, question)
            st.subheader("Answer")
            st.write(answer)

            with st.expander("Context / Source Excerpts"):
                for i, d in enumerate(docs_used):
                    st.markdown(f"**Source {i + 1}:**")
                    # show first 300 chars
                    excerpt = d.page_content[:300].replace("\n", " ")
                    st.write(excerpt + " …")

    else:
        st.info("Please upload one or more medical documents to begin.")

if __name__ == "__main__":
    main()
