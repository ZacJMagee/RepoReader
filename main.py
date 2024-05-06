import os
import tempfile
import re
import nltk
import uuid
import subprocess
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import DirectoryLoader, NotebookLoader, PythonLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

class SearchAndQueryManager:
    def __init__(self):
        nltk.download("punkt")

    def search_documents(self, query, index, documents, n_results=5):
        query_tokens = self.clean_and_tokenize(query)
        bm25_scores = index.get_scores(query_tokens)

        # Compute TF-IDF scores
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.clean_and_tokenize, lowercase=True, stop_words='english', use_idf=True, smooth_idf=True, sublinear_tf=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform([doc.page_content for doc in documents])
        query_tfidf = tfidf_vectorizer.transform([query])

        # Compute Cosine Similarity scores
        cosine_sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

        # Combine BM25 and Cosine Similarity scores
        combined_scores = bm25_scores * 0.5 + cosine_sim_scores * 0.5

        # Get unique top documents
        unique_top_document_indices = list(set(combined_scores.argsort()[::-1]))[:n_results]

        return [documents[i] for i in unique_top_document_indices]

    def format_documents(self, documents):
        numbered_docs = "\n".join([f"{i+1}. {os.path.basename(doc.metadata['source'])}: {doc.page_content}" for i, doc in enumerate(documents)])
        return numbered_docs

    def format_user_question(self, question):
        question = re.sub(r'\s+', ' ', question).strip()
        return question


    
    def ask_question(self, question, context):
        # Search for the top 5 most relevant documents based on the query
        relevant_docs = self.search_documents(question, context.index, context.documents, n_results=5)
        
        # Format the documents into a more concise context
        numbered_documents = self.format_documents(relevant_docs)
        question_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. Here are the most relevant documents:\n\n{numbered_documents}"

        # Run the LLM query with the formatted, concise context
        answer_with_sources = context.llm_chain.run(
            model=context.model_name,
            question=question,
            context=question_context,
            repo_name=context.repo_name,
            github_url=context.github_url,
            conversation_history=context.conversation_history,
            numbered_documents=numbered_documents,
            file_type_counts=context.file_type_counts,
            filenames=context.filenames
        )
        return answer_with_sources


class ApplicationController:
    def __init__(self):
        self.repository_manager = RepositoryManager()
        self.search_and_query_manager = SearchAndQueryManager()

    def main(self):
        github_url = input(WHITE + "Enter the GitHub URL of the repository: " + RESET_COLOR)
        repo_name = github_url.split("/")[-1]
        print(WHITE + "Cloning the repository..." + RESET_COLOR)
        
        with tempfile.TemporaryDirectory() as local_path:
            if self.repository_manager.clone_github_repo(github_url, local_path):
                index, documents, file_type_counts, filenames = self.repository_manager.load_and_index_files(local_path)
                
                if index is None:
                    print("No documents were found to index. Exiting.")
                    exit()

                print("Repository cloned. Indexing files...")
                llm = ChatOpenAI(model=model_name, temperature=0)

                # Create strings that will be used in the system message template
                file_type_counts_str = ', '.join(f"{k}: {v}" for k, v in file_type_counts.items())
                filenames_str = ', '.join(filenames)

                # Define the prompt with placeholders
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You are a helpful assistant for analyzing code repositories. "
                               f"Repository Name: {repo_name}, GitHub URL: {github_url}, "
                               f"Document Count: {len(documents)}, File Type Counts: {file_type_counts_str}, "
                               f"File Names: {filenames_str}"),
                    ("human", "{user_question}")
                ])
                chain = prompt | llm

                conversation_history = ""
                while True:
                    user_question = input(WHITE + "\nAsk a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
                    if user_question.lower() == 'exit()':
                        break
                    print(WHITE + 'Thinking...' + RESET_COLOR)

                    # Invoke with a dictionary that matches the variable names used in the template
                    full_response = chain.invoke({"user_question": user_question})

                    # Handle the response correctly assuming it is an individual AIMessage object
                    content = full_response.content
                    token_usage = full_response.response_metadata['token_usage']
                    print(GREEN + f'\nFULL RESPONSE\nContent: {content}\nToken Usage: Completion: {token_usage["completion_tokens"]}, Prompt: {token_usage["prompt_tokens"]}, Total: {token_usage["total_tokens"]}' + RESET_COLOR)
                    conversation_history += f"Question: {user_question}\nFull Response: Content: {content}, Token Usage: {token_usage}\n"
            else:
                print("Failed to clone the repository.")
