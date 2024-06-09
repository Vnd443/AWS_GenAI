
import json
import os
import sys
import boto3
import streamlit as st
import numpy as np

#from pypdf import PdfReader 

from langchain_community.embeddings import BedrockEmbeddings # We will be using Titan Embedding Models to Generate Embedding 
from langchain.llms.bedrock import Bedrock

from langchain.text_splitter import RecursiveCharacterTextSplitter # Data Ingestion 
#from langchain_community.document_loaders import PDFPlumberLoader,PyPDFLoader,PyPDFDirectoryLoader

from langchain_community.vectorstores import FAISS # Vector Embedding and Vector Store

from langchain.prompts import PromptTemplate # LLM Models
from langchain.chains import RetrievalQA
from langchain_community.retrievers import AmazonKendraRetriever

# Bedrock client
bedrock=boto3.client(service_name="bedrock-runtime",region_name = "us-east-1")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock) 
#bedrock_embeddings=BedrockEmbeddings(credentials_profile_name="default", model_id='amazon.titan-embed-text-v1',region_name='us-east-1')
kendra_client = boto3.client("kendra", region_name="us-east-1")
retriever = AmazonKendraRetriever(index_id='ad11e698-63e4-4896-9045-855f336c3c0a', client=kendra_client)
# Data Ingestion


# Vector Embedding and Vector store
def get_vector_store(docs):
    vectorstore_faiss=FAISS.load_local(docs,bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    
# Calling Titan LLM
def get_llm():
    llm=Bedrock(model_id="amazon.titan-text-lite-v1",client=bedrock,model_kwargs={"temperature": 0.5, "maxTokenCount": 300, "topP": 1})
                # model_kwargs={'max_gen_len':512})
    
    return llm



prompt_template="""

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm,vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k":3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']

def ret(query):
    index_id='ad11e698-63e4-4896-9045-855f336c3c0a'
    result = kendra_client.retrieve(
        IndexId = index_id,
        QueryText = query)
    ret=[]
    for retrieve_result in result["ResultItems"]:
        ret.append(str(retrieve_result["Content"]))
    return ret


def main():
    st.set_page_config("Chat PDF")
    st.header("Welcome to Personalized Assistant using AWS Bedrock💁")
    user_question = st.text_input("Ask a Question from the Retirement Services/EPFO")
    #with st.spinner("Loading..."):
    docs=ret(user_question)
    #print(docs)
    #get_vector_store(docs)  
    
    st.success('Ready to Use')

    if user_question:
        with st.spinner("Thinking..."):       
            #faiss_index=FAISS.load_local(,bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_llm()
            st.write(get_response_llm(llm,docs, user_question))
            st.success("Done")


if __name__ == "__main__":
    main()
