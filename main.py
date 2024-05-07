import os
import tempfile
import re
import nltk
import uuid
import subprocess
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, NotebookLoader, PythonLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from rank_bm25 import BM25Okapi

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WHITE = "\033[37m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0m"
model_name = "gpt-3.5-turbo"

nltk.download("punkt")

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

class RepositoryManager:
    def __init__(self):
        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def clone_github_repo(self, github_url, local_path):
        try:
            subprocess.run(['git', 'clone', github_url, local_path], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            return False

    def load_and_index_files(self, repo_path):
        extensions = {
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
            'editorconfig': TextLoader,
            'ipynb': NotebookLoader  # Special case for Jupyter notebooks
        }

        file_type_counts = {}
        documents_dict = {}

        for ext, LoaderClass in extensions.items():
            glob_pattern = f'**/*.{ext}'
            loader_kwargs = {'autodetect_encoding': True} if LoaderClass is TextLoader else {}
            try:
                if LoaderClass is NotebookLoader:
                    loader = LoaderClass(repo_path, include_outputs=True, max_output_length=20, remove_newline=True)
                else:
                    loader = DirectoryLoader(repo_path, glob=glob_pattern, loader_cls=LoaderClass, loader_kwargs=loader_kwargs, silent_errors=True)

                loaded_documents = loader.load()
                if loaded_documents:
                    file_type_counts[ext] = len(loaded_documents)
                    for doc in loaded_documents:
                        file_path = doc.metadata['source']
                        relative_path = os.path.relpath(file_path, repo_path)
                        file_id = str(uuid.uuid4())
                        doc.metadata['source'] = relative_path
                        doc.metadata['file_id'] = file_id

                        documents_dict[file_id] = doc
            except Exception as e:
                print(f"Error loading files with pattern '{glob_pattern}': {e}")
                continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

        split_documents = []
        for file_id, original_doc in documents_dict.items():
            split_docs = text_splitter.split_documents([original_doc])
            for split_doc in split_docs:
                split_doc.metadata['file_id'] = original_doc.metadata['file_id']
                split_doc.metadata['source'] = original_doc.metadata['source']

            split_documents.extend(split_docs)

        index = None
        if split_documents:
            tokenized_documents = [self.clean_and_tokenize(doc.page_content) for doc in split_documents]
            index = BM25Okapi(tokenized_documents)
        return index, split_documents, file_type_counts, [doc.metadata['source'] for doc in split_documents]


    def clean_and_tokenize(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\b(?:http|ftp)s?://\S+', '', text)
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        return nltk.word_tokenize(text)


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
            parent_splitter=self.parent_splitter
        )

    def add_documents(self, documents):
        self.retriever.add_documents(documents)

    def retrieve_documents(self, query):
        return self.retriever.invoke(query)

class SearchAndQueryManager:
    """Manages search and querying of documents, using EnhancedDocumentRetriever for document retrieval."""
    def __init__(self, document_retriever):
        self.document_retriever = document_retriever
        nltk.download("punkt")

    def search_documents(self, query, n_results=5):
        retrieved_docs = self.document_retriever.retrieve_documents(query)
        return retrieved_docs[:n_results]

    def format_documents(self, documents):
        numbered_docs = "\n".join([f"{i+1}. {os.path.basename(doc.metadata['source'])}: {doc.page_content[:500]}" for i, doc in enumerate(documents)])
        return numbered_docs


class ApplicationController:
    def __init__(self):
        self.repository_manager = RepositoryManager()

    def main(self):
        github_url = input(WHITE + "Enter the GitHub URL of the repository: " + RESET_COLOR)
        repo_name = github_url.split("/")[-1]
        print(WHITE + "Cloning the repository..." + RESET_COLOR)
        
        with tempfile.TemporaryDirectory() as local_path:
            if self.repository_manager.clone_github_repo(github_url, local_path):
                print("Repository cloned. Indexing files...")
                index, documents, file_type_counts, filenames = self.repository_manager.load_and_index_files(local_path)
                
                if not documents:
                    print("No documents were found to index. Exiting.")
                    exit()

                # Create the document retriever instance
                document_retriever = EnhancedDocumentRetriever(collection_name=repo_name)
                document_retriever.add_documents(documents)
                self.search_and_query_manager = SearchAndQueryManager(document_retriever)

                # Prepare the chat template
                file_type_counts_str = ', '.join(f"{k}: {v}" for k, v in file_type_counts.items())
                filenames_str = ', '.join(filenames)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You are a helpful assistant for analyzing code repositories. Repository Name: {repo_name}, GitHub URL: {github_url}, "
                               f"Document Count: {len(documents)}, File Type Counts: {file_type_counts_str}, File Names: {filenames_str}"),
                    ("human", "{user_question}")
                ])

                llm = ChatOpenAI(model=model_name, temperature=0)
                chain = prompt | llm
                conversation_history = ""
                while True:
                    user_question = input(WHITE + "\nAsk a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
                    if user_question.lower() == 'exit()':
                        break
                    print(WHITE + 'Thinking...' + RESET_COLOR)
                    # Search documents using the question
                    retrieved_docs = self.search_and_query_manager.search_documents(user_question, n_results=5)
                    formatted_docs = self.search_and_query_manager.format_documents(retrieved_docs)
                    print(GREEN + f"\nRETRIEVED DOCS\n{formatted_docs}" + RESET_COLOR)
                    
                    # Continue the conversation using the model
                    full_response = chain.invoke({"user_question": user_question})
                    content = full_response.content
                    token_usage = full_response.response_metadata['token_usage']
                    print(GREEN + f'\nFULL RESPONSE\nContent: {content}\nToken Usage: Completion: {token_usage["completion_tokens"]}, Prompt: {token_usage["prompt_tokens"]}, Total: {token_usage["total_tokens"]}' + RESET_COLOR)
                    conversation_history += f"Question: {user_question}\nFull Response: Content: {content}, Token Usage: {token_usage}\n"
            else:
                print("Failed to clone the repository.")
