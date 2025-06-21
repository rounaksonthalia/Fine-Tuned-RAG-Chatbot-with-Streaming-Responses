# Fine-Tuned-RAG-Chatbot-with-Streaming-Responses

Project Architecture & Flow

1. Document Ingestion: Load raw PDF files (e.g., AI Training document) using PyPDFLoader, then clean and format text (strip headers/footers) before further processing.


2. Chunking: Split cleaned documents into 100–300 word segments with a sentence-aware splitter (RecursiveCharacterTextSplitter), ensuring 20–50 word overlap for context continuity.


3. Embedding Generation: Use a semantic embedding model (Ollama) to convert each chunk into a vector. These embeddings capture semantic meaning for retrieval accuracy.


4. Vector Database: Index embeddings in FAISS for in-memory similarity search. FAISS provides fast k-NN lookups for streaming RAG applications.


5. Retriever: On each user query, perform a semantic search against FAISS to fetch the top‑k most relevant chunks.


6. Generator: Inject retrieved chunks and the user’s question into a prompt template. Use an instruction‑optimized open‑source LLM (e.g.llama-3-8b) via ChatGroq API for real‑time, token‑by‑token streaming of answers.


7. Streamlit Interface: Present a chat UI with:

Natural language input box

Live streaming of the LLM’s response

Expandable panel showing source chunks used

Sidebar details (model name, number of indexed chunks)

Reset button to clear chat history without rebuilding embeddings fileciteturn0file0.
