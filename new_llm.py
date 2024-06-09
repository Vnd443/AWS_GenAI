from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3

from langchain.chains import ConversationalRetrievalChain
import json
from langchain_community.retrievers import AmazonKendraRetriever
import streamlit as st

#Bedrock client
bedrock_client = boto3.client(service_name = "bedrock-runtime",region_name = "us-east-1")
kendra_client = boto3.client("kendra", region_name="us-east-1")
retriever = AmazonKendraRetriever(index_id='ad11e698-63e4-4896-9045-855f336c3c0a', client=kendra_client)

def lambda_handler(event, context):
    query=event
    #query = 'Defination rule 2'
    # query = event['query']

    question_generator_chain_template = """
    Human: Here is some chat history contained in the <chat_history> tags. If relevant, add context from the Human's previous questions to the new question. Return only the question. No preamble. If unsure, ask the Human to clarify. Think step by step.

    Assistant: Ok

    <chat_history>
    {chat_history}

    Human: {question}
    </chat_history>

    Assistant:
    """

    question_generator_chain_prompt = PromptTemplate.from_template(question_generator_chain_template)

    #Create template for asking the question of the given context.
    combine_docs_chain_template = """
    Human: You are a friendly, concise chatbot. Here is some context, contained in <context> tags. Answer this question as concisely as possible with no tags. Say I don't know if the answer isn't given in the context: {question}

    <context>
    {context}
    </context>

    Assistant:
    """
    combine_docs_chain_prompt = PromptTemplate.from_template(combine_docs_chain_template)
    llm = Bedrock(model_id="amazon.titan-text-lite-v1", client=bedrock_client, model_kwargs={"temperature": 0.5, "maxTokenCount": 300, "topP": 1})
    # RetrievalQA instance with custom prompt template
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=llm,
        retriever=retriever,
        return_source_documents=True,
        condense_question_prompt=question_generator_chain_prompt,
        combine_docs_chain_kwargs={"prompt": combine_docs_chain_prompt}
    )
    chat_history = []
    input_variables = {"question": query,"chat_history": chat_history}

    result = qa.invoke(input_variables)
    chat_history.append((query, result["answer"]))
    
    if(len(result['source_documents']) > 0):
        res=result['source_documents']
        document_titles=res[0].metadata['title']
        doc='Reference Doc name: '+document_titles
        response_text = result['answer']
        return response_text,doc
    else:
        response_text = "I don't know."
        doc='Please ask a Question from the Retirement Services'
        return response_text,doc

    
    #return {
    #    'statusCode': 200,
    #    'body': result_response
    # }



st.set_page_config("Centralized Knowledge Repository & Personalized Assistant")
st.header("Welcome to Personalized Assistant using AWS Bedrock")
query=st.text_input("Ask a Question from the Retirement Services")
if query:
    with st.spinner("Thinking..."):
        x,y=lambda_handler(query,'context')
        st.write(x)
        st.write(y)
        



