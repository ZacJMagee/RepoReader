import os
import sys
import tempfile
import uuid
import subprocess
from dotenv import load_dotenv
from langchain.indexes import SQLRecordManager, index
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.notebook import NotebookLoader
from langchain_community.document_loaders.python import PythonLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from anthropic import Anthropic

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

WHITE = "\033[37m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0m"
MODEL_NAME = "claude-3-5-sonnet-20240620"

class QuestionContext:
    def __init__(self, index, documents, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames):
        self.index = index
        self.documents = documents
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
            file_id = original_doc.metadata['file_id']
            split_docs = self.text_splitter.split_documents([original_doc])
            for split_doc in split_docs:
                split_doc.metadata['file_id'] = file_id
                split_doc.metadata['source'] = original_doc.metadata['source']
            split_documents.extend(split_docs)
        return split_documents

class RepositoryManager:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.extensions = {
            'txt': TextLoader, 'md': TextLoader, 'markdown': TextLoader, 'rst': TextLoader,
            'js': TextLoader, 'java': TextLoader, 'c': TextLoader,
            'cpp': TextLoader, 'cs': TextLoader, 'go': TextLoader, 'rb': TextLoader,
            'php': TextLoader, 'scala': TextLoader, 'html': TextLoader, 'htm': TextLoader,
            'xml': TextLoader, 'json': TextLoader, 'yaml': TextLoader, 'yml': TextLoader,
            'ini': TextLoader, 'toml': TextLoader, 'cfg': TextLoader, 'conf': TextLoader,
            'sh': TextLoader, 'bash': TextLoader, 'css': TextLoader, 'scss': TextLoader,
            'sql': TextLoader, 'ipynb': NotebookLoader
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
                loader_kwargs={'autodetect_encoding': True} if LoaderClass != PythonLoader else {},
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

        # Handle Python files separately
        py_loader = DirectoryLoader(
            self.repo_path,
            glob="**/*.py",
            loader_cls=PythonLoader,
            use_multithreading=True,
            show_progress=True
        )
        try:
            py_docs = py_loader.load()
            for doc in py_docs:
                doc_id = str(uuid.uuid4())
                doc.metadata['file_id'] = doc_id
                documents.append(doc)
        except Exception as e:
            print(f"Failed to load Python documents: {e}")

        return documents

class EnhancedDocumentRetriever:
    def __init__(self, collection_name):
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        self.store = InMemoryStore()
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model
        )
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            max_documents_per_query=5
        )

    def add_documents(self, documents):
        self.retriever.add_documents(documents)

    def retrieve_documents(self, query):
        return self.retriever.invoke(query)

class SearchAndQueryManager:
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
        self.repository_manager = None
        self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

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
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        vector_store = Chroma(collection_name=repo_name, embedding_function=embedding_model)
        record_manager = SQLRecordManager(namespace=f"{repo_name}_namespace", db_url="sqlite:///my_index.db")
        record_manager.create_schema()

        document_indexer = DocumentIndexer(vector_store, record_manager)
        document_indexer.index_documents(documents, cleanup_mode="incremental")

        document_retriever = EnhancedDocumentRetriever(collection_name=repo_name)
        document_retriever.add_documents(documents)

        return document_retriever

    def interactive_query(self, repo_name, document_retriever):
        system_prompt = f"You are a helpful assistant for analyzing the repository: {repo_name}. Ask me anything about it."
        
        while True:
            user_question = input(WHITE + "\nAsk a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
            if user_question.lower() == 'exit()':
                break
            print(WHITE + 'Thinking...' + RESET_COLOR)
            
            retrieved_docs = document_retriever.retrieve_documents(user_question)
            context = " ".join([doc.page_content for doc in retrieved_docs])

            messages = []
            if context:
                messages.append({"role": "user", "content": f"Context from the repository: {context}"})
                messages.append({"role": "assistant", "content": "I understand. How can I help you with this context?"})

            messages.append({"role": "user", "content": user_question})

            response = self.anthropic_client.messages.create(
                model=MODEL_NAME,
                system=system_prompt,
                messages=messages,
                max_tokens=1000
            )
            
            print(GREEN + f'\nFULL RESPONSE\nContent: {response.content[0].text}' + RESET_COLOR)

    def process_query(self, prompt, user_question):
        messages = [
            {"role": "user", "content": user_question}
        ]

        response = self.anthropic_client.messages.create(
            model=MODEL_NAME,
            system=prompt,
            messages=messages,
            max_tokens=1000
        )

        print(GREEN + f'\nFULL RESPONSE\nContent: {response.content[0].text}' + RESET_COLOR)

if __name__ == "__main__":
    app_controller = ApplicationController()
    app_controller.main()
