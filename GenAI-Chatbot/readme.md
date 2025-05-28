# GenAI-Chatbot

A Streamlit-based chatbot that allows you to upload PDF documents, extract their content, and ask questions about them using local open-source language models and embeddings.

---

## Features

- **PDF Upload:** Upload one or more PDF files via the sidebar.
- **Text Extraction:** Extracts and splits text from uploaded PDFs into manageable chunks.
- **Embeddings:** Uses HuggingFace's `all-MiniLM-L6-v2` model for generating embeddings.
- **Vector Store:** Stores embeddings in a local FAISS vector database for efficient similarity search.
- **Question Answering:** Uses the `google/flan-t5-base` model (run locally) to answer user questions based on the uploaded documents.
- **No OpenAI API Required:** All models run locally or via HuggingFace, so no OpenAI credits or API keys are needed.

---

## Semantic Analysis Libraries Used

This application uses the following libraries for semantic analysis:

- **sentence-transformers**: Generates semantic vector embeddings for text chunks.
- **faiss-cpu**: Stores and performs similarity search on embeddings for semantic retrieval.
- **langchain** and **langchain_community**: Orchestrate the workflow for chunking, embedding, retrieval, and question answering.
- **transformers**: Loads and runs the local language model (`google/flan-t5-base`) for answering questions based on retrieved semantic content.

---

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/GenAI-Chatbot.git
   cd GenAI-Chatbot
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

   > **Note:** If you encounter errors related to Rust during installation, install Rust from [https://rustup.rs/](https://rustup.rs/) and retry.

---

## Usage

1. **Run the Streamlit app:**
   ```sh
   streamlit run chatbot.py
   ```

2. **In your browser:**
   - Upload one or more PDF files using the sidebar.
   - Ask questions about the content in the main input box.

---

## Notes

- The app uses local models (`all-MiniLM-L6-v2` for embeddings, `google/flan-t5-base` for Q&A). These will be downloaded automatically the first time you run the app.
- For best performance, use a machine with at least 8GB RAM.
- If you want to use a different model, change the model names in `chatbot.py`.

---

## License

MIT License

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [FAISS](https://github.com/facebookresearch/faiss)