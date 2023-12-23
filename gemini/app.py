# install the following:
# - streamlit
# - google-generativeai
# - python-dotenv

from dotenv import load_dotenv
load_dotenv() ## loading all the env variables

import streamlit as st 
import os
import google.generativeai as genai 

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

## function to load gemini model
model = genai.GenerativeModel("gemini-pro")

def get_gemini_response(question):
    response = model.generate_content(question)
    return response.text


## initialize the streamlit app

st.set_page_config(page_title="Q&A DEMO")
st.header("Gemini LLM Application")
input = st.text_input("Input: ", key = "input")
submit = st.button("Ask the question")

## when submit is clicked 

if submit:
    response = get_gemini_response(input)
    st.subheader("The response is ")
    st.write(response)