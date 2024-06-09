import boto3
import pprint

kendra = boto3.client("kendra",region_name="us-east-2")

# Provide the index ID
index_id = 'db651331-76f3-4bfd-adbf-331684edea61'
# Provide the query text
query ='Defination rule 2'
# You can retrieve up to 100 relevant passages
# You can paginate 100 passages across 10 pages, for example
page_size = 10
page_number = 10

result = kendra.retrieve(
        IndexId = index_id,
        QueryText = query)

print("\nRetrieved passage results for query: " + query + "\n")        

for retrieve_result in result["ResultItems"]:

    print("-------------------")
    print("Title: " + str(retrieve_result["DocumentTitle"]))
    print("URI: " + str(retrieve_result["DocumentURI"]))
    print("Passage content: " + str(retrieve_result["Content"]))
    print("------------------\n\n")