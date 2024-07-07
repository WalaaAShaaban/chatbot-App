from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from src.embedding import embedding
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter

class ChainPDF():

    google_api_key = "AIzaSyBFbjwM2aC-P17T2vsRgpKbDJ8O4gYPSow"
    llm = GoogleGenerativeAI(model="gemini-pro",google_api_key=google_api_key,temperature=0)

    
    def load_chunk_persist_pdf(self) -> Chroma:
        loader = PyPDFLoader('input/Introduction to Machine Learning with Python.pdf')
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        chunked_documents = text_splitter.split_documents(pages)
        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=embedding(),
            persist_directory="input/db"
        )
        vectordb.persist()
        return vectordb
   

    def create_pdf_chain(self):
        vector_db = self.load_chunk_persist_pdf()
        chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=vector_db.as_retriever())
        return chain
      

    

        

        
