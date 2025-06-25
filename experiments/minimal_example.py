# %%
import os
import sys
from pathlib import Path
from langchain.document_loaders import PyPDFLoader

# Change to project root directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from backend.RAG import EmbeddedPDF  # noqa: E402

# %%
DATA_PATH = "backend/data/books"
book = "Harry-Potter-and-the-Philosophers-Stone"
loader = PyPDFLoader(DATA_PATH + "/" + book + ".pdf")
pages = loader.load()

# %%
pdf_embedder = EmbeddedPDF()
pdf_embedder.embed_pdf(pages)
# %% [markdown]
### Generate character analysis for a specific character
# %%
response = pdf_embedder.generate_character_analysis("Hermione Granger", full_book=True)
print(response)
# %% [markdown]
### Perform a semantic search for character-related context (up to current page)
# %%
# Before we meet Hermione
pdf_embedder.set_current_page(75)
response = pdf_embedder.generate_character_analysis("Hermione Granger", full_book=False)
print(response)

# %%
# After we meet Hermione
pdf_embedder.set_current_page(100)
response = pdf_embedder.generate_character_analysis("Hermione Granger", full_book=False)
print(response)

# %% [markdown]
### Check for newly introduced characters on the specified page
# %%
response = pdf_embedder.check_page_for_characters(pages[77].page_content)
print(response)
