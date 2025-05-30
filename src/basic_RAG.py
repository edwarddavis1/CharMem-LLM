# %%
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os 
import shutil

# %%
# LOAD THE DATA

DATA_PATH = "../data/books"
book = "Harry-Potter-and-the-Philosophers-Stone"
loader = PyPDFLoader(DATA_PATH + "/" + book + ".pdf")
pages = loader.load()

# %%
# CHUNK THE TEXT

chunk_size = 1000
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True,
)
chunks = splitter.split_documents(pages)

# %%
# USE VECTOR DATABASE TO EMBED EACH OF THE CHUNKS

# Free model from hugging face
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database by embedding each of the chunks using the
#  specified embedding model
CHROMA_PATH = "chroma"

# Remove previous database if making a new one
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Create vector database
db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)


# %%
# RAG

# Load the database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Search the database
query_text = "Who is Hermione Granger?"
results = db.similarity_search_with_relevance_scores(query_text, k=3)

# # Filter
# if len(results) == 0 or results[0][1] < 0.5:
#     print("Unable to find good results")

# View the retrieval
retrieval = "\n\n---\n\n".join([page.page_content for page, _ in results])
print(retrieval)

# %%
# CREATE RESPONSE
PROMPT_TEMPLATE = """
You are a helpful book assistant. Given the following excerpts from a novel, provide the user information about a specified character as clearly and concisely as possible, using only the provided text.

You will provide an answer in three distinct paragraphs to provide information about the following:
1. A summary of the character.
2. Where we first met the character (including the page number and how they were introduced)
3. Some recent events involving the character (recent, i.e. higher page numbers).

It is important that your answers are formatted like in the following examples.

Example 1:
Character: Harry Potter
Answer:
Harry Potter is...

We first meet Harry Potter on page XX, where he was XX...

Recently, Harry Potter has...

Example 2:
Character: Orsen Kovacs
[In the case where there is no relevant context given about this character]
Answer:
We have not met a character named Orsen Kovacs.

Context:
{context}

Character: {query}

Answer:"""


prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=retrieval, query=query_text)

# %%
from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="mistral-7b-instruct.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    verbose=True,
)
