import streamlit as st
import yaml
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pandas as pd
import io

load_dotenv()

def initialize_session_state():
    if 'config' not in st.session_state:
        with open('config.yml', 'r') as file:
            st.session_state.config = yaml.safe_load(file)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "valid_user" not in st.session_state:
        st.session_state.valid_user = False

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

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

@st.cache_data
def login(namespace):
    if namespace in st.secrets['users']:
        return True, namespace
    else:
        return False, None

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_excel(excel_file):
    df = pd.read_excel(excel_file)
    return df.to_string()

def process_files(files, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for file in files:
        if file.name.endswith('.pdf'):
            raw_text = extract_text_from_pdf(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            raw_text = extract_text_from_excel(file)
        else:
            continue  # Skip unsupported file types
        texts.extend(text_splitter.split_text(raw_text))
    
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

def main():
    st.set_page_config(page_title="Aayu.ai")
    initialize_session_state()

    llm, embeddings, custom_prompt = get_resources()

    with st.sidebar:
        namespace = st.text_input('Login', value="")
        start = st.button('Start', type='primary')
        if start:
           st.session_state['valid_user'], namespace = login(namespace)

        if st.session_state['valid_user']:
            uploaded_files = st.file_uploader("Upload PDF or Excel files", type=["pdf", "xlsx", "xls"], accept_multiple_files=True)
            if uploaded_files:
                with st.spinner("Processing files..."):
                    st.session_state.vectorstore = process_files(uploaded_files, embeddings)
                st.success("Files processed successfully!")

    st.markdown("# Aayu.ai")
    st.caption('Health Expert')
    if not st.session_state['valid_user']:
        st.warning('Login to begin')
    else:
        if st.button('Clear chat history', type='primary'):
            st.session_state.memory.clear()
            st.session_state.messages.clear()
            st.success('Chat history cleared!')

        with st.chat_message("assistant"):
            st.write("Hello 👋 I'm here to provide information based on the documents you've uploaded and general health knowledge. How can I assist you today?")

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
                        combine_docs_chain_kwargs={"prompt": custom_prompt}
                    )
                    response = qa_chain({"question": prompt})
                    answer = response['answer']
                else:
                    answer = "Please upload some PDF or Excel documents first so I can assist you based on their content. I can also provide general health information if you have any questions."

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.markdown(answer)

if __name__ == '__main__':
    main()