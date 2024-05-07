import os
import tempfile
import uuid
import uuid
import subprocess
from dotenv import load_dotenv
from langchain.indexes import SQLRecordManager, index
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, NotebookLoader, PythonLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WHITE = "\033[37m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0m"
model_name = "gpt-3.5-turbo"

class QuestionContext:
    def __init__(self, index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames):
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model_name = model_name
        self.repo_name = repo_name
        self.github_url = github_url
        self.conversation_history = conversation_history
        self.file_type_counts = file_type_counts
        self.filenames = filenames


class DocumentIndexer:
    def __init__(self, vector_store, record_manager):
        self.vector_store = vector_store
        self.record_manager = record_manager
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def index_documents(self, documents, cleanup_mode="incremental"):
        split_documents = self.split_and_prepare(documents)
        index(split_documents, self.record_manager, self.vector_store, cleanup=cleanup_mode, source_id_key="source")

    
    def split_and_prepare(self, documents):
        split_documents = []
        for original_doc in documents:
            file_id = original_doc.metadata['file_id']  # Assuming 'file_id' is stored in metadata
            split_docs = self.text_splitter.split_documents([original_doc])
            for split_doc in split_docs:
                split_doc.metadata['file_id'] = file_id  # Ensure file ID is maintained
                split_doc.metadata['source'] = original_doc.metadata['source']
            split_documents.extend(split_docs)
        return split_documents

class RepositoryManager:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.extensions = {
            'txt': TextLoader,
            'md': TextLoader,
            'markdown': TextLoader,
            'rst': TextLoader,
            'py': PythonLoader,
            'js': TextLoader,
            'java': TextLoader,
            'c': TextLoader,
            'cpp': TextLoader,
            'cs': TextLoader,
            'go': TextLoader,
            'rb': TextLoader,
            'php': TextLoader,
            'scala': TextLoader,
            'html': TextLoader,
            'htm': TextLoader,
            'xml': TextLoader,
            'json': TextLoader,
            'yaml': TextLoader,
            'yml': TextLoader,
            'ini': TextLoader,
            'toml': TextLoader,
            'cfg': TextLoader,
            'conf': TextLoader,
            'sh': TextLoader,
            'bash': TextLoader,
            'css': TextLoader,
            'scss': TextLoader,
            'sql': TextLoader,
            'ipynb': NotebookLoader  # Special case for Jupyter notebooks
        }

    def clone_github_repo(self, github_url, local_path):
        try:
            subprocess.run(['git', 'clone', github_url, local_path], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            return False

    def load_documents(self):
        documents = []
        for ext, LoaderClass in self.extensions.items():
            loader = DirectoryLoader(
                self.repo_path,
                glob=f"**/*.{ext}",
                loader_cls=LoaderClass,
                loader_kwargs={'autodetect_encoding': True},
                use_multithreading=True,
                show_progress=True
            )
            try:
                docs = loader.load()
                for doc in docs:
                    doc_id = str(uuid.uuid4())
                    doc.metadata['file_id'] = doc_id
                    documents.append(doc)
            except Exception as e:
                print(f"Failed to load documents for {ext}: {e}")
        return documents


class EnhancedDocumentRetriever:
    """Handles efficient document retrieval by managing indexed document chunks and their parent documents."""
    def __init__(self, collection_name, dimensions=3072):
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        self.store = InMemoryStore()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", dimensions=dimensions)
        )
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            max_documents_per_query=5  # Limiting the number of documents returned per query
        )

    def add_documents(self, documents):
        self.retriever.add_documents(documents)

    def retrieve_documents(self, query):
        return self.retriever.invoke(query)

class SearchAndQueryManager:
    """Manages search and querying of documents, using EnhancedDocumentRetriever for document retrieval."""
    def __init__(self, document_retriever):
        self.document_retriever = document_retriever

    def search_documents(self, query, n_results=5):
        retrieved_docs = self.document_retriever.retrieve_documents(query)
        return retrieved_docs[:n_results]

    def format_documents(self, documents):
        numbered_docs = "\n".join([f"{i+1}. {os.path.basename(doc.metadata['source'])}: {doc.page_content[:500]}" for i, doc in enumerate(documents)])
        return numbered_docs


class ApplicationController:
    def __init__(self):
        # This no longer takes repo_path as a parameter at initialization.
        self.repository_manager = None

    def main(self):
        github_url = input(WHITE + "Enter the GitHub URL of the repository: " + RESET_COLOR)
        repo_name = github_url.split("/")[-1]
        print(WHITE + "Cloning the repository..." + RESET_COLOR)

        with tempfile.TemporaryDirectory() as local_path:
            self.repository_manager = RepositoryManager(local_path)
            if self.repository_manager.clone_github_repo(github_url, local_path):
                print("Repository cloned. Loading and indexing files...")
                documents = self.repository_manager.load_documents()

                if not documents:
                    print("No documents were found to index. Exiting.")
                    return

                document_retriever = self.setup_document_retriever(documents, repo_name)
                self.interactive_query(repo_name, document_retriever)
            else:
                print("Failed to clone the repository.")

    def setup_document_retriever(self, documents, repo_name):
        # Setup vector store and record manager
        vector_store = Chroma(collection_name=repo_name, embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072))
        record_manager = SQLRecordManager(namespace=f"{repo_name}_namespace", db_url="sqlite:///my_index.db")
        record_manager.create_schema()

        # Index documents
        document_indexer = DocumentIndexer(vector_store, record_manager)
        document_indexer.index_documents(documents, cleanup_mode="incremental")

        # Setup document retriever
        document_retriever = EnhancedDocumentRetriever(collection_name=repo_name, dimensions=3072)
        document_retriever.add_documents(documents)

        return document_retriever

    
    
    def interactive_query(self, repo_name, document_retriever):
        # Define the system's initial prompt/message
        system_prompt = f"You are a helpful assistant for analyzing the repository: {repo_name}. Ask me anything about it."
        
        # Initialize the llm model
        llm = ChatOpenAI(model=model_name, temperature=0)
        
        while True:
            user_question = input(WHITE + "\nAsk a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
            if user_question.lower() == 'exit()':
                break
            print(WHITE + 'Thinking...' + RESET_COLOR)
            
            # Retrieve document chunks relevant to the query if necessary
            retrieved_docs = document_retriever.retrieve_documents(user_question)
            context = " ".join([doc.page_content for doc in retrieved_docs])  # Concatenate content for context

            # Create messages as per the required format
            messages = [
                ("system", system_prompt),
                ("human", user_question)
            ]

            # If there's additional context from documents, add it as another system message
            if context:
                messages.insert(1, ("system", context))

            # Invoke the model with the formatted messages
            response = llm.invoke(messages)
            
            # Output the response content
            print(GREEN + f'\nFULL RESPONSE\nContent: {response.content}' + RESET_COLOR)

    def process_query(self, prompt, llm, user_question):
        # Here, we need to invoke the chain correctly
        full_response = prompt.invoke({"user_question": user_question}) | llm
        print(GREEN + f'\nFULL RESPONSE\nContent: {full_response.content}' + RESET_COLOR)

if __name__ == "__main__":
    app_controller = ApplicationController()
    app_controller.main()
