import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="SKT Chatbot", page_icon=":robot_face:", layout="wide")

# Upload PDF files
st.header("SKT Chatbot")

with st.sidebar:
    st.title("Documents")
    file = st.file_uploader("Upload PDF files and start asking questions", type=["pdf"], accept_multiple_files=True)

all_chunks = []

# Extract text from PDF files
if file is not None:
    for f in file:
        if f is not None:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            # st.write(text)
            # Break down the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
            st.write(chunks)

# Create embeddings for the text chunks
if file is not None and all_chunks:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Create a vector store from the text chunks - FAISS
    vector_store = FAISS.from_texts(all_chunks, embeddings)

    # Get user query
    user_query = st.text_input("Ask a question about Meteorology:")

    # Perform similarity search
    if user_query:
        # Search the vector store for similar chunks
        results = vector_store.similarity_search(user_query)

        # Define LLM model for question answering
        # Create a local pipeline (download model if not present)
        hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        # Display the results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=results, question=user_query)
        st.write(response)