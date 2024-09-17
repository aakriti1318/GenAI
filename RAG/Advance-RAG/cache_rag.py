import fitz
import numpy as np
import cv2
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from functools import lru_cache

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        # Extract text
        text += page.get_text()
        
        # Extract tables
        tables = page.find_tables()
        for table in tables:
            df = table.to_pandas()
            text += "\nTable:\n" + df.to_string() + "\n"
        
        # Extract images and perform OCR
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert image bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Perform OCR
            ocr_text = pytesseract.image_to_string(image)
            text += f"\nImage {img_index + 1} OCR:\n{ocr_text}\n"
    
    return text

@lru_cache(maxsize=100)
def cached_embedding(text):
    return OpenAIEmbeddings().embed_query(text)

class CachedFAISS(FAISS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_fn = cached_embedding

def create_vectorstore(texts, embeddings):
    return CachedFAISS.from_texts(texts, embeddings)

def setup_qa_chain(vectorstore, llm, memory, custom_prompt):
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

def rag_without_caching(query, pdf_file, openai_api_key):
    raw_text = extract_text_from_pdf(pdf_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.5,
        openai_api_key=openai_api_key
    )
    
    custom_prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant specializing in ... topics...
        # Rest of the prompt
    """)
    
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    
    qa_chain = setup_qa_chain(vectorstore, llm, memory, custom_prompt)
    
    return qa_chain({"question": query})

def rag_with_caching(query, pdf_file, openai_api_key):
    raw_text = extract_text_from_pdf(pdf_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = create_vectorstore(texts, embeddings)
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.5,
        openai_api_key=openai_api_key
    )
    
    custom_prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant specializing in ... topics...
        # Rest of the prompt
    """)
    
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    
    qa_chain = setup_qa_chain(vectorstore, llm, memory, custom_prompt)
    
    return qa_chain({"question": query})

# Example usage
pdf_file = ""
openai_api_key = ""
query = "Compare GPT-1 and BERT"

result_without_caching = rag_without_caching(query, pdf_file, openai_api_key)
result_with_caching = rag_with_caching(query, pdf_file, openai_api_key)

print("Result without caching:", result_without_caching)
print("Result with caching:", result_with_caching)
