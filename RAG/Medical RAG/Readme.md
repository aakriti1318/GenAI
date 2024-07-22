# Aayu.ai Health Assistant

Aayu.ai is an AI-powered health information assistant that leverages Retrieval-Augmented Generation (RAG) to provide accurate and contextual responses to health-related queries. This application processes various types of input data, including PDFs, Excel files, and web pages, to enhance its knowledge base and provide more informed answers.

## Features

- User authentication system
- Processing of multiple data sources:
  - PDF documents
  - Excel spreadsheets
  - Web pages (via URL)
- Conversational AI interface
- Retrieval-Augmented Generation for accurate responses
- Streamlit-based user interface

## Architecture

The Aayu.ai Health Assistant is built using the following key components:

1. **Frontend**: Streamlit
2. **Language Model**: OpenAI's GPT model (via LangChain)
3. **Embedding Model**: OpenAI's embedding model
4. **Vector Store**: FAISS (Facebook AI Similarity Search)
5. **Data Processing**: PyMuPDF, Pandas, BeautifulSoup4
6. **Memory**: ConversationBufferMemory from LangChain

### Data Flow

1. User uploads files or provides a URL
2. Data is processed and converted to text
3. Text is split into chunks
4. Chunks are embedded and stored in FAISS vector store
5. User asks a question
6. Question is used to retrieve relevant context from vector store
7. Retrieved context and question are sent to LLM
8. LLM generates a response
9. Response is displayed to the user

## Data Processing

### PDF Processing

PDF documents are processed using the PyMuPDF library (imported as `fitz`). The process involves:

1. Opening the PDF file
2. Iterating through each page
3. Extracting text content from each page
4. Combining extracted text into a single string

```python
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text
```

### Excel Processing

Excel files are processed using the Pandas library. The process includes:

1. Reading the Excel file into a Pandas DataFrame
2. Converting the DataFrame to a string representation

```python
def extract_text_from_excel(excel_file):
    df = pd.read_excel(excel_file)
    return df.to_string()
```

### Web Page Processing

Web pages are processed using the Requests library for fetching content and BeautifulSoup for parsing HTML. The process involves:

1. Sending a GET request to the provided URL
2. Parsing the HTML content using BeautifulSoup
3. Extracting all text content from the parsed HTML

```python
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()
```

## Retrieval-Augmented Generation (RAG)

The RAG process in Aayu.ai works as follows:

1. Extracted text from all sources (PDFs, Excel files, web pages) is split into smaller chunks using RecursiveCharacterTextSplitter.
2. These chunks are embedded using OpenAI's embedding model.
3. Embeddings are stored in a FAISS vector store for efficient similarity search.
4. When a user asks a question, it's used to retrieve the most relevant chunks from the vector store.
5. Retrieved chunks, along with the conversation history and the user's question, are sent to the language model.
6. The language model generates a response based on this context and its own knowledge.

This approach allows the AI to provide responses that are grounded in the specific information from the uploaded documents and web pages, while also leveraging its general knowledge.

## Setup and Usage
To install the MultiPDF Chat App, please follow these steps:

1. Clone the repository to your local machine. 
    ```
    git clone https://github.com/aayu.ai -b aakriti
    ```
2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```
3. Obtain an API key from HuggingFace and add it to the `.env` file in the project directory.
    ```
    OPENAI_API_KEY=your_secret_api_key
    ```
4. Install the following:
    ```
    brew install tesseract
    brew install tesseract-lang 
    ```

-----
To use the Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the API keys to the `.env` file.

2. Rename `streamlit` Directory to `.streamlit`

3. Run the `app.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

## Architecture

<img width="789" alt="User Flow Diagram" src="files/architectural_diagram.png">
