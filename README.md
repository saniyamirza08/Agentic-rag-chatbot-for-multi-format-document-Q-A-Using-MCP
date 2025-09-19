# Agentic RAG Chatbot for Multi-Format Document QA using MCP

This project is an agent-based Retrieval-Augmented Generation (RAG) chatbot that can answer user questions using uploaded documents of various formats. The architecture follows an agentic structure and incorporates the Model Context Protocol (MCP) for communication between agents.

## Features

-   **Multi-Format Document Support:** Upload and process PDF, PPTX, CSV, DOCX, and TXT/Markdown files.
-   **Agentic Architecture:** The system is built with three distinct agents:
    -   `IngestionAgent`: Parses and preprocesses documents.
    -   `RetrievalAgent`: Handles embedding and semantic retrieval.
    -   `LLMResponseAgent`: Forms the final LLM query and generates the answer.
-   **Model Context Protocol (MCP):** Agents communicate using structured MCP messages, ensuring clear and traceable interactions.
-   **Vector Store and Embeddings:** Uses OpenAI embeddings and an in-memory vector store for efficient retrieval.
-   **Chatbot Interface:** A user-friendly interface built with Streamlit that allows for document uploads, multi-turn questions, and viewing responses with source context.

## Architecture

The application follows a multi-agent RAG architecture:

1.  **UI (Streamlit):** The user uploads documents and asks questions.
2.  **IngestionAgent:** When a document is uploaded, the `IngestionAgent` parses the file, extracts the text, and splits it into chunks.
3.  **RetrievalAgent:** The `RetrievalAgent` takes the text chunks, generates embeddings using OpenAI's API, and stores them in an in-memory vector store. When a query is received, it retrieves the most relevant chunks.
4.  **LLMResponseAgent:** The `LLMResponseAgent` receives the retrieved chunks and the user's query, forms a prompt, and sends it to the LLM to generate a final answer.
5.  **MCP:** All communication between agents is handled through structured MCP messages, which are printed to the terminal for real-time monitoring.

## Tech Stack

-   **Language:** Python
-   **UI:** Streamlit
-   **LLM:** OpenAI
-   **Embeddings:** OpenAI
-   **Document Parsing:** PyPDF2, python-docx, python-pptx, pandas
-   **Vector Store:** In-memory (NumPy)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```

5.  **Run the application:**
    ```bash
    streamlit run main2.py
    ```

## Usage

1.  **Upload Documents:** Use the file uploader in the sidebar to upload your documents.
2.  **Ask Questions:** Enter your questions in the text input box and press Enter.
3.  **View Responses:** The chatbot will display the answer along with the source chunks used to generate it.
4.  **Monitor MCP Messages:** The MCP messages exchanged between the agents will be printed to the terminal in real-time.