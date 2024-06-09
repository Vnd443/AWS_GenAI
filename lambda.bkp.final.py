import os
import json
import boto3
import pdfplumber
import pandas as pd
import PyPDF2
from io import BytesIO
from botocore.exceptions import ClientError
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
#from langchain.vectorstores.pgvector import PGVector
from langchain_community.retrievers import AmazonKendraRetriever
#from langchain.indexes import index

from langchain.chains import RetrievalQA

''  


def lambda_handler(event, context):
    # Get content of uploaded object
    s3=boto3.resource(service_name='s3',region_name = "us-east-1")
  
    # Set up client for Amazon Bedrock

    bedrock_client=boto3.client( service_name = "bedrock-runtime",region_name = "us-east-1")
    bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock_client) 
    kendra_client=boto3.client("kendra",region_name = "us-east-1")
    retriever=AmazonKendraRetriever(index_id='ad11e698-63e4-4896-9045-855f336c3c0a',client=kendra_client,top_k=3)
    #rep=retriever.get_relevant_documents('Defination rule 2')
    #result=kendra_client.query(IndexId="cdfd8e45-3011-4114-b867-963d082ee758",QueryText='defination rule 2')
    # Set up client for Amazon Kendra
    #kendra=boto3.client("kendra")
    #query='tell about cricket'
    #print(retriever)
    #retriever=kendra.retrieve(IndexId=,QueryText=query)
    #retriever=AmazonKendraRetriever(index_id="9a3a51a4-02e6-451f-b6d0-c59a0ef7c760",top_k=5)
    
   
    llm=Bedrock(model_id="amazon.titan-text-lite-v1",client=bedrock_client,model_kwargs={"temperature": 0.5, "maxTokenCount": 300, "topP": 1})
   
    
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

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
        
    condense_qa_template = """
    Given the following conversation and a follow up question, rephrase the follow up question 
    to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)
      #standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)
    #qa=ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, condense_question_prompt=standalone_question_prompt, return_source_documents=True, combine_docs_chain_kwargs={"prompt":PROMPT},verbose=True)
    #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,return_source_documents=True,chain_type_kwargs={"prompt": PROMPT})
     
    #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    def get_response_llm(llm,query,history=[]):
        
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        answer=qa({"query":query})
        return answer['result']
    query='Who is the CEO of IBM'
    chat_history = []
    k=get_response_llm(llm,query,chat_history)
    print(k,chat_history)
    #print(qa({"query":'Defination rule 2'}))
    
    #response = qa(query)
    print()
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, qa_prompt=PROMPT, return_source_documents=True)
    def run_chain(chain, prompt, history=[]):
      answer=qa({"query":prompt})
      return answer['result']
      #return chain({"question": prompt, "chat_history": history})
    chat_history = []
    #print(chat_history)
    result = run_chain(qa, 'Defination rule 2', chat_history)
    print(result)
    #or d in result['source_documents']:
    #    d.metadata['source']

    
    print(chat_history)

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
        
    #docs=data_ingestion('idea2test')
    print('docs loaded')
    #get_vector_store(docs)
    print('vector stored')
    def load_faiss_index_from_s3(bucket_name, key):
        s3 = boto3.client('s3')

        # Retrieve the FAISS index file from S3
        response = s3.get_object(Bucket=bucket_name, Key=key)
        faiss_index_content = response['Body'].read()

        # Load the FAISS index from the content
        faiss_index = FAISS.load_s3(faiss_index_content, bedrock_embeddings, allow_dangerous_deserialization=True)

        return faiss_index
      
    #faiss_index=load_faiss_index_from_s3('idea2test','faiss_index')
    #print('faiss loaded')
    #llm=get_llm()
    #user_question='Defination rule 2'
    #get_response_llm(llm, faiss_index, user_question)


def build_chain():


  s3=boto3.resource(service_name='s3',region_name = "us-east-1")

  # Set up client for Amazon Bedrock

  
  bedrock_client=boto3.client( service_name = "bedrock-runtime",region_name = "us-east-1")
  bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock_client) 
  retriever=AmazonKendraRetriever(index_id="cdfd8e45-3011-4114-b867-963d082ee758",top_k=5)
    
  #retriever = AmazonKendraRetriever(index_id=kendra_index_id,top_k=5,region_name=region)


  prompt_template =""" Human: Use the following pieces of context to provide a 
  concise answer to the question at the end but usse atleast summarize with 
  500 words with detailed explaantions. If you don't know the answer, 
  just say that you don't know, don't try to make up an answer.
  <context>
  {context}
  </context

  Question: {question}

  Assistant:"""
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )

  condense_qa_template = """
  Given the following conversation and a follow up question, rephrase the follow up question 
  to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:"""
  standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)
  llm=Bedrock(model_id="amazon.titan-text-lite-v1",client=bedrock_client,model_kwargs={"temperature": 0.5, "maxTokenCount": 300, "topP": 1})
  qa=ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        condense_question_prompt=standalone_question_prompt, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt":PROMPT},
        verbose=True)

  # qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, qa_prompt=PROMPT, return_source_documents=True)
  print(qa)
  


#print(build_chain())
lambda_handler('a','b')


    #faiss_index=FAISS.load_local('faiss_index',bedrock_embeddings, allow_dangerous_deserialization=True)
