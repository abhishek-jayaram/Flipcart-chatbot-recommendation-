import pandas as pd

 
MAX_TEXT_LENGTH=1000  # Maximum num of text characters to use
 
def auto_truncate(val):
 
    """Truncate the given text."""
 
    return val[:MAX_TEXT_LENGTH]
    
# Load Product data and truncate long text fields
 
all_prods_df = pd.read_csv("flipkart_com-ecommerce_sample.csv", converters={
 
    'description': auto_truncate,
 
    'product_specifications': auto_truncate,
 
    'product_name': auto_truncate,
    
    'product_category_tree': auto_truncate,
 
})

# %%
# Replace empty strings with None and drop
 
all_prods_df['product_specifications'].replace('', None, inplace=True)
 
all_prods_df.dropna(subset=['product_specifications'], inplace=True)
 
# Reset pandas dataframe index
 
all_prods_df.reset_index(drop=True, inplace=True)

# %%
# Num products to use (subset)
NUMBER_PRODUCTS = 1000

# Get the first 1000 products
product_metadata = ( 
    all_prods_df
     .head(NUMBER_PRODUCTS)
     .to_dict(orient='index')
)
 
# Check one of the products
product_metadata[0]

# %%
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Redis
from langchain_community.vectorstores.redis import RedisVectorStore
# set your openAI api key as an environment variable
# os.environ['OPENAI_API_KEY'] = "your_api_key_here"
 
# data that will be embedded and converted to vectors
texts = [
    v['product_name'] for k, v in product_metadata.items()
]

# product metadata that we'll store along our vectors
metadatas = list(product_metadata.values())
 
# we will use OpenAI as our embeddings provider
embedding = OpenAIEmbeddings()
 
index_name = "products"
redis_url = "redis://localhost:6379"
# %%
vectorstore = RedisVectorStore.from_texts(
    texts=texts,
    embedding=embedding,
    metadatas=metadatas,
    index_name=index_name,
    redis_url=redis_url
)