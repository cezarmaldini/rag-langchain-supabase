import os
from dotenv import load_dotenv
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

load_dotenv()

docs = [
    Document(page_content="foo", metadata={"id": 1}),
]

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

SUPABASE_URL = os.getenv('SUPABASE_URL')

SUPABASE_API_KEY = os.getenv('SUPABASE_API_KEY')

supabase_client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase_client,
    table_name="documents",
    query_name="match_documents",
    chunk_size=500,
)