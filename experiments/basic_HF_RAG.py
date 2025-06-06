# %%
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import shutil
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os
import sys
from pathlib import Path

# Change to project root directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from backend.config import MODEL_ID, EMBEDDING_MODEL_ID

# %%
# Load environment variables
load_dotenv()

HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
# %%
# LOAD THE DATA

DATA_PATH = "backend/data/books"
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

embedding_function = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL_ID, huggingfacehub_api_token=HF_API_TOKEN
)

# Create vector database by embedding each of the chunks using the
#  specified embedding model
CHROMA_PATH = "chroma"

# Remove previous database if making a new one
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Create vector database
db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)

# Load the database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# %%
# RAG

# Search the database
query_text = "Hermione Granger"
results = db.similarity_search_with_relevance_scores(query_text, k=50)

# # Filter
# if len(results) == 0 or results[0][1] < 0.5:
#     print("Unable to find good results")

# View the retrieval
# retrieval = "\n\n---\n\n".join([page.page_content for page, _ in results])
retrieval = "\n\n---\n\n".join(
    [
        f"[Page {page.metadata.get('page', 'N/A')}]\n{page.page_content}"
        for page, _ in results
    ]
)
# retrieval = ""
# %%
# CREATE RESPONSE
PROMPT_TEMPLATE = """
You are a helpful book assistant. Given the following excerpts from a novel, provide the user information about a specified character as clearly and concisely as possible, using only the provided text.

You will provide an answer in three distinct paragraphs to provide information about the following:
1. A summary of the character.
2. Where we first met the character (including the page number and how they were introduced)
3. Some recent events involving the character (recent, i.e. higher page numbers).

If there is not enough evidence that we have met this character, you must say that we have not met the character.

It is important that your answers are formatted like in the following examples.

Example 1:
Character: Harry Potter
Answer:
Harry Potter is... [What sets them apart from other characters, their role in the story, etc.]

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
# LOAD LLM

client = InferenceClient(api_key=HF_API_TOKEN)

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": prompt}],
    # max_tokens=1000,
    temperature=0.7,
)

bot_response = response.choices[0].message.content

print(bot_response)

# %%
