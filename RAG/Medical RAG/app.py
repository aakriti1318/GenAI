import streamlit as st
import yaml
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np

load_dotenv()

class CustomConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        input_str = inputs['question']
        output_str = outputs['answer'] if 'answer' in outputs else str(outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

def initialize_session_state():
    if 'config' not in st.session_state:
        with open('config.yml', 'r') as file:
            st.session_state.config = yaml.safe_load(file)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = CustomConversationBufferMemory(return_messages=True, memory_key="chat_history")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

@st.cache_resource
def get_resources():
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    llm = ChatOpenAI(
        model_name=st.session_state.config['llm'],
        temperature=0.5,
        openai_api_key=openai_api_key
    )
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    custom_prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant specializing in health-related topics. Use the following pieces of context and the conversation history to answer the question in layman's terms. If you don't have specific information about a particular case or personal medical details, provide general, factual information related to the topic.

    Remember:
    1. Do not give any personal medical advice or diagnosis.
    2. If asked for a diagnosis, say that you're not allowed to give any diagnosis and recommend consulting a healthcare professional.
    3. Provide factual, general information about health topics when possible.
    4. If you're unsure or the question is outside of the provided context and general health knowledge, admit that you don't have enough information to answer accurately.

    Context: {context}

    Chat History: {chat_history}

    Question: {question}

    Answer:""")
    
    return llm, embeddings, custom_prompt

# def extract_text_from_pdf(pdf_file):
#     text = ""
#     pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
#     for page_num in range(len(pdf_document)):
#         page = pdf_document.load_page(page_num)
#         text += page.get_text()
#     return text

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

def extract_text_from_excel(excel_file):
    df = pd.read_excel(excel_file)
    return df.to_string()

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def process_files(files, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    metadatas = []
    for file in files:
        if file.name.endswith('.pdf'):
            raw_text = extract_text_from_pdf(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            raw_text = extract_text_from_excel(file)
        else:
            continue  # Skip unsupported file types
        chunks = text_splitter.split_text(raw_text)
        texts.extend(chunks)
        metadatas.extend([{"source": file.name} for _ in chunks])
    
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore

def process_url(url, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    raw_text = extract_text_from_url(url)
    texts = text_splitter.split_text(raw_text)
    metadatas = [{"source": url} for _ in texts]
    
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore

def main():
    st.set_page_config(page_title="Medical Bot")
    initialize_session_state()

    llm, embeddings, custom_prompt = get_resources()

    st.markdown("#Medical Bot")
    st.caption('Health Expert')

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload PDF or Excel files", type=["pdf", "xlsx", "xls"], accept_multiple_files=True)
        url = st.text_input("Or enter a web page URL")
        
        if uploaded_files:
            with st.spinner("Processing files..."):
                st.session_state.vectorstore = process_files(uploaded_files, embeddings)
            st.success("Files processed successfully!")
        
        if url:
            with st.spinner("Processing URL..."):
                url_vectorstore = process_url(url, embeddings)
                if st.session_state.vectorstore:
                    st.session_state.vectorstore.merge_from(url_vectorstore)
                else:
                    st.session_state.vectorstore = url_vectorstore
            st.success("URL processed successfully!")

    if st.button('Clear chat history', type='primary'):
        st.session_state.memory.clear()
        st.session_state.messages.clear()
        st.success('Chat history cleared!')

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter question"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner():
            if st.session_state.vectorstore:
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm,
                    retriever=st.session_state.vectorstore.as_retriever(),
                    memory=st.session_state.memory,
                    combine_docs_chain_kwargs={"prompt": custom_prompt},
                    return_source_documents=True
                )
                response = qa_chain.invoke({"question": prompt})
                answer = response['answer']
                source_documents = response['source_documents']
                
                # Extract source information
                sources = set([doc.metadata['source'] for doc in source_documents if 'source' in doc.metadata])
                source_info = f"\n\nSources: {', '.join(sources)}" if sources else ""
                
                answer_with_sources = f"{answer}{source_info}"
            else:
                answer_with_sources = "Please upload some PDF or Excel documents, or provide a web page URL first so I can assist you based on their content. I can also provide general health information if you have any questions."

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": answer_with_sources})

        with st.chat_message("assistant"):
            st.markdown(answer_with_sources)

if __name__ == '__main__':
    main()
