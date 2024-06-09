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
#from langchain.chains import ConversationalRetrievalChain,VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
#from langchain.vectorstores.pgvector import PGVector
#from langchain.indexes import index


'''
for i in s3.buckets.all():
    print(i)
#s3.Bucket('bucketname').upload_file(Filename='Filename',Key='model_key')
'''  


def lambda_handler(event, context):
    # Get content of uploaded object
    s3=boto3.resource(service_name='s3',region_name = "us-east-1")
  
    # Set up client for Amazon Bedrock

    
    bedrock_client=boto3.client( service_name = "bedrock-runtime",region_name = "us-east-1")
    bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock_client) 
    
    # Set up client for Amazon Kendra
    kendra=boto3.client("kendra")
    query='tell about cricket'
    
    #retriever=kendra.retrieve(IndexId=,QueryText=query)
    #retriever=AmazonKendraRetriever(index_id="9a3a51a4-02e6-451f-b6d0-c59a0ef7c760",top_k=5)
    
    # Data Ingestion
    def data_ingestion(bucket_name):
        s3 = boto3.client('s3')


        # List objects in the specified S3 bucket
        response = s3.list_objects_v2(Bucket=bucket_name)


        # Initialize an empty list to store the PDF content
        documents = []

        # Iterate over each object in the response
        for obj in response.get('Contents', []):
            # Retrieve the key (document ID) of the object
            document_key = obj['Key']
            #print(document_key)
            #print(document_key)
            # Check if the object is a PDF file (you may need to adjust this condition)
            if document_key.endswith('.pdf'):
                # Retrieve the PDF document content from S3
              response = s3.get_object(Bucket=bucket_name, Key=document_key)
              #fileObj = s3.get_object(Bucket=bucketname, Key=filename)
        # reading botocore stream
              content = response['Body'].read()

              # Create a PDF file reader object
              pdf_reader = PyPDF2.PdfReader(BytesIO(content))
              #print(pdf_reader)
              #text_splitter=RecursiveCharacterTextSplitter(separators=['\n\n','\n',' ',''],chunk_size=1000,chunk_overlap=500)
              #print(text_splitter.split_documents(pdf_reader))
              text_splitter=RecursiveCharacterTextSplitter(separators=['\n\n','\n',' ',''],chunk_size=100,chunk_overlap=50)
              page_contents = ''

              # Iterate over each page and extract text
              for page_num in range(len(pdf_reader.pages)):
                  page = pdf_reader.pages[page_num]
                  page_contents+=page.extract_text()
                  #p=text_splitter.split_text(text=page_contents)
                  #print(p)

              documents.append(page_contents)
              # Pass the list of page contents to the split_documents function
        #print(type(''.join(page_contents)))
        #docs = text_splitter.split_text(','.join(documents))
        #print(docs)
              #print(pdf_content)
        #print(s3.get_object(Bucket=bucket_name, Key='GPF_Rules.pdf')['Body'].read())
            # Append the (pdfPDF content to the list of documents
            #documents.append(text)
        
        return ','.join(documents)
    class PageContent:
      def __init__(self, content,metadata=None):
          self.page_content = content
          self.metadata = metadata
    def get_vector_store(docs):

      #document_objects = [PageContent(page) for page in docs]
      # Create the vector store from the PDF content
      #print('doccs',document_objects)
      vectorstore_faiss=FAISS.from_texts(docs, bedrock_embeddings)
      
      print('search')
      
      # Save the vector store to S3
      vectorstore_faiss.save_s3(bucket='idea2test', key='faiss_index')
      print('saved')
      return vectorstore_faiss

    def get_llm():
      llm=Bedrock(model_id="amazon.titan-text-lite-v1",client=bedrock_client,model_kwargs={"temperature": 0.5, "maxTokenCount": 300, "topP": 1})
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

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
        
    #standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

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
        
    docs=data_ingestion('idea2test')
    print('docs loaded')
    get_vector_store(docs)
    print('vector stored')
    def load_faiss_index_from_s3(bucket_name, key):
        s3 = boto3.client('s3')

        # Retrieve the FAISS index file from S3
        response = s3.get_object(Bucket=bucket_name, Key=key)
        faiss_index_content = response['Body'].read()

        # Load the FAISS index from the content
        faiss_index = FAISS.load_s3(faiss_index_content, bedrock_embeddings, allow_dangerous_deserialization=True)

        return faiss_index
      
    faiss_index=load_faiss_index_from_s3('idea2test','faiss_index')
    print('faiss loaded')
    llm=get_llm()
    user_question='Defination rule 2'
    get_response_llm(llm, faiss_index, user_question)

lambda_handler('a','b')

    #faiss_index=FAISS.load_local('faiss_index',bedrock_embeddings, allow_dangerous_deserialization=True)
