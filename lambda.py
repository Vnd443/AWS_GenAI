import json
import boto3
from botocore.exceptions import ClientError
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.retrievers import AmazonKendraRetriever

def lambda_handler(event, context):
  query=event
  #query = 'Defination rule 2'
  # query = event['query']

  # Set up client for Amazon network
  bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
  bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
  kendra_client = boto3.client("kendra", region_name="us-east-1")
  #retriever = AmazonKendraRetriever(index_id='ad11e698-63e4-4896-9045-855f336c3c0a', client=kendra_client,)
  def kendraq1(query):
    index_id='ad11e698-63e4-4896-9045-855f336c3c0a'
    result = kendra_client.retrieve(
        IndexId = index_id,
        QueryText = query, PageNumber=1,
        PageSize=15)
    result1=''
    for i in result:
      print(str(i["Content"]))
      result1=str(i["Content"])
      break
    #return invokeLLM(query, result1)
  def invokeLLM(question, kendra_response):
    bedrock=boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    #bedrock=Bedrock(model_id="amazon.titan-text-lite-v1", client=bedrock_client, model_kwargs={"temperature": 0.5, "maxTokenCount": 300, "topP": 1})
    accept = 'application/json'
    contentType = 'application/json'
    prompt_data = f"""\n\nHuman:    
Answer the following question to the best of your ability based on the context provided.
Provide an answer and provide sources and the source link to where the relevant information can be found. Include this at the end of the response
Do not include information that is not relevant to the question.
Only provide information based on the context provided, and do not make assumptions
Only Provide the source if relevant information came from that source in your answer
Use the provided examples as reference
###
Question: {question}

Context: {kendra_response}

###

\n\nAssistant:

"""
    body = json.dumps({"prompt": prompt_data,
                       "max_tokens_to_sample": 8191,
                       "temperature": 0,
                       "top_k": 250,
                       "top_p": 0.5,
                       "stop_sequences": []
                       })

    #PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    response = bedrock.invoke_model(body=body,
                                    modelId="amazon.titan-text-lite-v1",
                                    accept=accept,
                                    contentType=contentType)
    
    response_body = json.loads(response.get('body').read())
    answer = response_body.get('completion')
    # returning the answer as a final result, which ultimately gets returned to the end user
    return answer
     
  
  llm = Bedrock(model_id="amazon.titan-text-lite-v1", client=bedrock_client, model_kwargs={"temperature": 0.5, "maxTokenCount": 300, "topP": 1})

  prompt_template = """
  Human: Use the following pieces of context to provide a 
  concise answer to the question at the end but usse atleast summarize with 
  500 words with detailed explaantions. If you don't know the answer, 
  just say that you don't know, don't try to make up an answer out of retriever.
  <context>
  {context}
  </context

  Question: {question}

  Assistant:"""

  PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
  k=kendraq1(query)
  print(k)

  def get_response_llm(llm, query, history=[]):
      qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=k,return_source_documents=False, chain_type_kwargs={"prompt": PROMPT})
      answer = qa.invoke({"query": query})  # Using invoke method instead of __call__
      #print(answer)
      return answer['result']

  chat_history = []

  #result = get_response_llm(llm, query, chat_history)

  return result
  #print(result)
  #result_response = json.loads(result)

  #return {
  #    'statusCode': 200,
  #    'body': result_response
 # }

input_query=input('Please Enter your Question: ')

print(lambda_handler(input_query,'b'))