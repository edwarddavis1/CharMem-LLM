# %%
import os
import sys
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
import pandas as pd
from tqdm import tqdm

# Change to project root directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from backend.RAG import EmbeddedPDF  # noqa: E402

# %%
# Embed the book

DATA_PATH = "backend/data/books"
book = "Harry-Potter-and-the-Philosophers-Stone"
loader = PyPDFLoader(DATA_PATH + "/" + book + ".pdf")
pages = loader.load()

pdf_embedder = EmbeddedPDF()
pdf_embedder.embed_pdf(pages)
# %%
# Load the character data
df = pd.read_csv(
    "experiments/first_meet_evaluation_data/HP_character_analysis_manual.csv"
)
data = df[["Character", "First_Appearance"]]

# %% [markdown]
### Evaluate the RAG first meet analysis
# %%

results = []
for character in tqdm(data["Character"]):
    # LLM page number
    llm_page = pdf_embedder.get_character_first_mention(character)

    # Get actual page number
    actual_page = data[data["Character"] == character]["First_Appearance"].iloc[0]

    # Store results
    results.append(
        {
            "Character": character,
            "Actual_Page": actual_page,
            "LLM_Page": llm_page,
            "Correct": llm_page == actual_page if llm_page else False,
        }
    )

    print(
        f"{character}: Actual={actual_page}, LLM={llm_page}, Correct={llm_page == actual_page if llm_page else False}"
    )

# Convert to DataFrame for easy analysis
results_df = pd.DataFrame(results)
print("\nSummary:")
print(results_df)

# %%
