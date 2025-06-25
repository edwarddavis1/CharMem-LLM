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
response = pdf_embedder.generate_character_analysis("Hermione Granger")
print(response)
# %% [markdown]
###

# %% [markdown]
### Check for newly introduced characters on the specified page
# %%
response = pdf_embedder.check_new_characters(1)
print(response)
