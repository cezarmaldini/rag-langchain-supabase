from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from fastembed import TextEmbedding
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.docstore.document import Document

from langchain.embeddings import HuggingFaceEmbeddings

from supabase import create_client
import uuid
import os

class Ingestor:
    def __init__(self, file_converter: DocumentConverter, chunker, dense_model, vector_store):
        self.file_converter = file_converter
        self.chunker = chunker
        self.dense_model = dense_model
        self.vector_store = vector_store

    def read_doc(self, file: str):
        return self.file_converter.convert(file)

    def doc_to_chunk_iter(self, doc):
        return self.chunker.chunk(dl_doc=doc.document)

    def chunk_to_document(self, chunk):
        # Criamos o objeto Document do LangChain
        return Document(
            page_content=chunk.text,
            metadata={"id": str(uuid.uuid4())}
        )

    def file_to_documents(self, file: str):
        doc = self.read_doc(file)
        return [self.chunk_to_document(chunk) for chunk in self.doc_to_chunk_iter(doc)]

    def ingest_documents(self, documents):
        self.vector_store.add_documents(documents)

    def ingest_file(self, file: str):
        documents = self.file_to_documents(file)
        self.ingest_documents(documents)

def new_ingestor(collection_name):
    # Conexão com Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Embedding model compatível com LangChain
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    # Criação do VectorStore no Supabase
    vector_store = SupabaseVectorStore(
        client=supabase_client,
        table_name=collection_name,
        embedding=embedding_model
    )

    # Configuração do chunker
    chunker = HybridChunker(
        tokenizer="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        max_tokens=768,
        merge_peers=True
    )

    return Ingestor(
        file_converter=DocumentConverter(),
        chunker=chunker,
        dense_model=embedding_model,
        vector_store=vector_store
    )

def ingest_files(ingestor, data_folder="data/articles"):
    files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.endswith(".md")
    ]
    for f in files:
        ingestor.ingest_file(f)

def main():
    collection_name = "documents_collection"
    ingestor = new_ingestor(collection_name)
    ingest_files(ingestor)
    return ingestor

if __name__ == "__main__":
    main()