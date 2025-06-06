# CharMem: AI-Powered Character Memory Assistant

> Never lose track of those minor characters again!

CharMem is an intelligent chat application that helps readers keep track of characters in books and novels. Using retireval augmented generation (RAG), it provides instant character summaries, first meeting details, and recent events with precise page references.

## 🌟 Key Features

-   **📄 PDF Upload & Processing**: Upload any PDF book for instant character analysis
-   **🔍 Smart Character Search**: Ask about any character and get comprehensive summaries
-   **🤖 AI-Powered Responses**: Powered by Qwen-235B model via Hugging Face Inference API
-   **💬 Real-time Chat Interface**: WebSocket-based chat with typing indicators
-   **📍 Page References**: Get exact page numbers for character introductions and events
-   **🎨 Modern UI**: Responsive dark-themed interface with smooth animations

## 🚀 Quick Start

### Prerequisites

-   Python 3.11 or 3.12
-   Hugging Face API token

### Installation

#### Option 1: Using UV (Recommended)

1. **Install UV ([see instructions](https://docs.astral.sh/uv/getting-started/installation/))**

2. **Clone and setup**

    ```bash
    git clone https://github.com/edwarddavis1/CharMem-LLM.git
    cd CharMem
    uv sync
    ```

3. **Run the application**
    ```bash
    uv run run.py
    ```

#### Option 2: Using Standard Python/Pip

1. **Clone the repository**

    ```bash
    git clone https://github.com/edwarddavis1/CharMem-LLM.git
    cd CharMem
    ```

2. **Create virtual environment**

    ```bash
    python -m venv .venv

    # On Windows
    .venv\Scripts\activate

    # On macOS/Linux
    source .venv/bin/activate
    ```

3. **Install dependencies**

    ```bash
    pip install .
    ```

4. **Run the application**
    ```bash
    python run.py
    ```

#### Final Steps (Both Options)

1. **Generate a Hugging Face API token**

    - Go to [Hugging Face](https://huggingface.co/docs/hub/en/security-tokens) and create a user access token.
    - Copy the token for use in the next step.

2. **Set up environment variables**
   Create a `.env` file in the root directory:

    ```bash
    HUGGINGFACE_API_TOKEN=your_huggingface_api_token_here
    ```

3. **Open your browser**
   Navigate to `http://localhost:8000`

## 💡 How It Works

1. **Upload a PDF**: Click "Upload PDF" to process your book
2. **Ask about characters**: Type character names in the chat
3. **Get instant insights**: Receive character summaries, first meetings, and recent events
4. **Continue reading**: All responses include page references to help you navigate

## 🔧 Minimal Code Example

Here's how to use CharMem programmatically:

```python
from backend.RAG import EmbeddedPDF
from langchain.document_loaders import PyPDFLoader

# Load your PDF
loader = PyPDFLoader("path/to/your/book.pdf")
pages = loader.load()

# Create embedder and process PDF
pdf_embedder = EmbeddedPDF()
pdf_embedder.embed_pdf(pages)

# Query character information
response = pdf_embedder.generate_character_analysis("Character Name")
print(response)
```

## 🛠️ Technology Stack

-   **Backend**: FastAPI with WebSocket support
-   **AI/ML**: Hugging Face Inference API, LangChain, ChromaDB
-   **Frontend**: Vanilla JavaScript with modern CSS
-   **Vector Embeddings**: sentence-transformers/all-MiniLM-L6-v2
-   **Language Model**: Qwen-235B-A22B

## 📝 Example Output

```
You: Who is Hermione Granger?

CharMem: Hermione Granger is an exceptionally intelligent and bookish student at Hogwarts,
known for her dedication to academic excellence and strict adherence to rules.
Initially portrayed as overbearing and socially awkward, she is eager to prove her
worth and often corrects others’ mistakes. Her character evolves from a rule-
follower to someone willing to bend regulations for the greater good, balancing
her thirst for knowledge with growing camaraderie.

Hermione is first introduced on **Page 77** during the train ride to Hogwarts. She
enters Harry and Ron’s compartment, introducing herself as a Muggle-born witch who
has memorized all their textbooks and is eager to learn magic.

In later chapters (e.g., **Page 182, 194, 216–221**), Hermione plays critical
roles in key plot points:
- She accompanies Harry and Ron into the Forbidden Forest to investigate unicorn
deaths, interacting nervously with centaurs like Ronan.
- She helps Harry confront Snape and protect the Philosopher’s Stone, calming him
with a hug before he faces danger and later explaining Dumbledore’s reasoning for
allowing Harry to proceed.
- During a confrontation in the dungeon chambers, she solves the potion riddle,
choosing to stay behind to ensure Harry’s safe passage.
- At the end-of-year feast (**Page 219**), Dumbledore awards her 50 points for
“cool logic in the face of fire,” recognizing her pivotal role in saving the Stone.
- On the final pages (**221**), she expresses concern for Harry’s well-being
during summer break, highlighting her growth into a supportive, empathetic friend.
Her academic prowess is underscored when she finishes top of the year
(**Page 220**), cementing her as both a scholar and a hero.
```

## 🏗️ Project Structure

```
CharMem/
├── app/
│   ├── static/          # CSS and JavaScript files
│   └── templates/       # HTML templates
├── backend/
│   ├── RAG.py          # RAG implementation and PDF processing
│   └── config.py       # Model configuration
├── experiments/        # Development and testing scripts
├── main.py            # FastAPI application
├── run.py             # Application runner
└── pyproject.toml     # Dependencies and project config
```

## 🔮 Features Coming Soon...

-   **Reading Progress Tracking**: Track your current page to avoid spoilers
-   **Enhanced Search**: Plot summaries and thematic analysis
-   **In-Character Responses**: Chat with characters as if they were real
