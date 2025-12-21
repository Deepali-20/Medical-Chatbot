from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls= PyPDFLoader
    )

    documents = loader.load()
    return documents
def filter_to_minimal_docs(docs:List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of document objects
    containg only source in metadata and the orignal page_content."""
    minimal_docs:List[Document]=[]
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source":src}
            )
        )
    return minimal_docs
# split the documents into smaller chunks
def text_split(minimal_doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap= 20

    )
    texts_chunk= text_splitter.split_documents(minimal_doc)
    return texts_chunk
## download the embedding file
def download_hugging_face_embeddings():
    """
    Download and return the HuggingFace embedings model.
    """
    model_name = "sentence-transformers/all-miniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        
    )
    return embeddings
embeddings = download_hugging_face_embeddings()

