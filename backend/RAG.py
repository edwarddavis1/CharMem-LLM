import io
import os

import PyPDF2
from dotenv import load_dotenv
from fastapi import UploadFile
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from backend.config import EMBEDDING_MODEL_ID, MODEL_ID

load_dotenv()
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


async def file_to_langchain_doc(pdf: UploadFile) -> list[Document]:
    """
    Converts a FastAPI UploadFile object to a list of langchain Document objects.

    Args:
        pdf (UploadFile): The PDF file uploaded via FastAPI.

    Returns:
        list[Document]: A list of langchain Document objects, each representing a page in the PDF.
    """
    content = await pdf.read()

    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
    pages = []

    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        doc = Document(
            page_content=page_text,
            metadata={
                "source": pdf.filename,
                "page": page_num,
                "total_pages": len(pdf_reader.pages),
            },
        )
        pages.append(doc)

    return pages


def chunk_langchain_pages(
    pages: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 500,
    add_start_index: bool = True,
) -> list[Document]:
    """
    Splits a list of langchain Document objects into smaller chunks.

    Args:
        pages (list[Document]): List of Document objects to chunk
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        add_start_index (bool): Whether to add start index to metadata

    Returns:
        list[Document]: List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=add_start_index,
    )
    chunks = splitter.split_documents(pages)
    return chunks


class EmbeddedPDF:
    """Manages PDF processing, vector database, and character analysis."""

    def __init__(
        self,
        num_return_chunks=50,
        chunk_size=1000,
    ):
        self.db: Chroma | None = None
        self.embedding_function = None
        self._total_pages = 0
        self._current_page = 0

        self.chunk_size = chunk_size
        self.num_return_chunks = num_return_chunks

        self.client = InferenceClient(api_key=HF_API_TOKEN)
        self.embedding_function = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL_ID,
            huggingfacehub_api_token=HF_API_TOKEN,
        )

    def set_current_page(self, page: int):
        self._current_page = page

    def set_total_pages(self, total_pages: int):
        self._total_pages = total_pages

    def embed_pdf(self, pages: list[Document]) -> dict:
        """Embed a list of langchain Document objects into a vector database."""
        try:
            # Chunk the content of the pdf
            chunks = chunk_langchain_pages(pages)

            # Embed the chunks to create the vector database
            self.db = Chroma.from_documents(chunks, self.embedding_function)
            self.set_total_pages(len(pages))

            return {
                "success": True,
                "pages": len(pages),
                "message": "PDF processed successfully",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def semantic_search(
        self, character_name: str, k: int = 50, full_book: bool = False
    ) -> str:
        """Search for character-related context in the database."""

        if self.db is None:
            raise ValueError("No PDF has been processed yet")

        if full_book:
            page_limit = self._total_pages
        else:
            page_limit = self._current_page

        # Search entire book
        results = self.db.similarity_search_with_relevance_scores(character_name, k=k)

        # Filter results to only include pages within the page_limit
        filtered_results = [
            (page, score)
            for page, score in results
            if page.metadata.get("page", 0) < page_limit
        ]
        retrieval = "\n\n---\n\n".join(
            [
                f"[Page {page.metadata.get('page', 'N/A') + 1}]\n{page.page_content}"
                for page, _ in filtered_results
            ]
        )

        return retrieval

    def generate_character_analysis(
        self, character_name: str, full_book: bool = False
    ) -> str:
        """Generate character analysis using the LLM."""
        context = self.semantic_search(
            character_name, k=self.num_return_chunks, full_book=full_book
        )

        PROMPT_TEMPLATE = """
        You are a helpful book assistant. Given the following excerpts from a novel, provide the user information about a specified character as clearly and concisely as possible, using only the provided text.

        You will provide an answer in three distinct paragraphs to provide information about the following:
        1. A summary of the character.
        2. Where we first met the character (including the page number and how they were introduced)
        3. Some recent events involving the character (recent, i.e. higher page numbers).

        Remember to keep your answer as concise as possible and relevant to the provided context.

        If there is not enough evidence that we have met this character, you must say "We have not met this character" only - do not say anything else other than this exact statement.

        Context:
        {context}

        Character: {query}

        Answer:"""

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, query=character_name)

        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return response.choices[0].message.content or ""

    def has_documents(self) -> bool:
        """Check if the database has any documents."""
        return self.db is not None

    def check_page_for_characters(self, page: str) -> str:
        """Check for newly introduced characters on the current page."""
        if self.db is None:
            raise ValueError("No PDF has been processed yet")

        PROMPT_TEMPLATE = """
        You are a helpful book assistant. Given the following page from a novel, check if any new characters are introduced on this page. If there are new characters, provide their names only - do not produce any text other than the character names, separated by commas.

        Page: {pdf_page}
        """

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(pdf_page=page)

        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return response.choices[0].message.content or ""

    def get_character_first_mention(
        self,
        character_name: str,
    ) -> int | None:
        """Get the page number where a character is first mentioned."""
        context = self.semantic_search(
            character_name, k=self.num_return_chunks, full_book=True
        )

        PROMPT_TEMPLATE = """
        You are a helpful book assistant. Given the following excerpts from a novel, find the page number where the specified character is first mentioned.

        Context:
        {context}

        Character: {query}

        Answer with either 'PAGE: <page_number>' if the character is mentioned or 'Not found' if the character is not mentioned.
        """

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, query=character_name)

        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        # Extract page number from response
        text = response.choices[0].message.content or ""

        if "PAGE:" in text:
            try:
                page_number = int(text.split("PAGE:")[1].strip())
                return page_number
            except ValueError:
                return None
        elif "Not found" in text:
            print(f"Character '{character_name}' not found in the provided context.")
            return None
        else:
            # If the response format is unexpected, log it and return None
            print(f"Unexpected response format: {text}")
            return None
