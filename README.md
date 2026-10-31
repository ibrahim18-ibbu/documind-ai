# Universal PDF Assistant (Ollama only)

A local Retrieval-Augmented Generation (RAG) app that answers questions from PDFs on any subject.

## API keys
This project does **not** use Gemini, OpenAI, or any cloud API key. Text generation runs through Ollama on your computer. The embedding model `all-MiniLM-L6-v2` is downloaded the first time it is used; no API key is needed.

## Requirements
- Windows 10/11
- Python 3.10 or newer (Python 3.12 is a safe choice for compatibility)
- Ollama for Windows

## Setup on Windows

1. Install Ollama from https://ollama.com/download/windows
2. Open a new PowerShell terminal in this project folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If Python 3.12 is not installed, use a Python version supported by your installed ML packages.

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Download and test the local language model:

   ```powershell
   ollama run llama3.2:3b
   ```

   Wait for the model to download. At the `>>>` prompt, test it with a question, then type `/bye` to exit. Ollama must be running in the background while using the app.

6. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

7. In the browser, upload one or more PDFs, click **Process PDFs**, and ask questions.

## Troubleshooting

- **`ollama` is not recognized:** Restart the terminal after installing Ollama. If it still fails, verify the Ollama installation and PATH. The usual Windows executable directory is `%LOCALAPPDATA%\Programs\Ollama`.
- **Ollama connection error:** Open the Ollama desktop app, or run `ollama serve` in another terminal if the service is not already running.
- **Model missing:** Run `ollama pull llama3.2:3b`, then retry.
- **No text found:** This version extracts selectable PDF text. Scanned/image-only PDFs need OCR, which is not included.
- **First run is slow:** The sentence-transformer embedding model downloads the first time. It is not an API key or a paid API.
- **Python install errors:** Use a Python version supported by PyTorch and sentence-transformers. If using Python 3.14 and installation fails, try Python 3.12.

## What it does
- Multiple PDF uploads
- Text extraction with filename and page tracking
- Chunking and local semantic search
- Local answer generation using Ollama
- Retrieved source passages shown in the UI
- Follow-up chat history passed to the model

## Limitations
The answer quality depends on PDF text extraction and retrieval. Scanned PDFs, complex tables, and image-based diagrams may require OCR or specialized extraction. The model can still make mistakes, so verify important information against the cited pages.
