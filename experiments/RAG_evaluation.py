# %%
import os
import sys
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
import pandas as pd
import re
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
df = pd.read_csv("experiments/first_meet_evaluation_data/HP_character_analysis.csv")
data = df[["Character", "First_Appearance"]]


def extract_first_meeting_page(text: str) -> int | None:
    """
    Extract page number from the '2.' section (first meeting paragraph).

    Args:
        text: The LLM response text

    Returns:
        Page number from first meeting section, or None if not found
    """
    # Split text into lines and look for "2." section
    lines = text.split("\n")

    # Find the line that starts with "2."
    for i, line in enumerate(lines):
        if line.strip().startswith("2."):
            # Look in this line and the next few lines for a page number
            search_text = line
            # Include next 3 lines to capture multi-line paragraphs
            for j in range(1, 4):
                if i + j < len(lines):
                    search_text += " " + lines[i + j]

            # Look for page patterns: (Page 77), **Page 77**, Page 77
            patterns = [r"\(Page\s+(\d+)\)", r"\*+Page\s+(\d+)\*+", r"Page\s+(\d+)"]

            for pattern in patterns:
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    return int(match.group(1))

    return None


# %% [markdown]
### Evaluate the RAG first meet analysis
# %%

results = []
for character in tqdm(data["Character"]):
    response = pdf_embedder.generate_character_analysis(character, full_book=True)

    # Get actual page from data
    actual_page = data[data["Character"] == character]["First_Appearance"].iloc[0]

    # Extract LLM's predicted page
    llm_page = extract_first_meeting_page(response)

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
