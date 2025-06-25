"""
Tests for the RAG.py module.
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import UploadFile
from langchain_core.documents import Document

from backend.RAG import EmbeddedPDF, chunk_langchain_pages, file_to_langchain_doc
from tests.testing_setup import (
    MockChroma,
    MockHuggingFaceEmbeddings,
    MockInferenceClient,
    MockPyPDF2Reader,
)


@pytest.fixture
def sample_pdf_upload():
    """Create a mock UploadFile for testing."""
    content = b"Sample PDF content for testing"
    mock_file = Mock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=content)
    mock_file.filename = "test_book.pdf"
    return mock_file


@pytest.fixture
def sample_documents():
    """Create sample Document objects for testing."""
    return [
        Document(
            page_content="Harry Potter is a young wizard who lives with his aunt and uncle.",
            metadata={"source": "test_book.pdf", "page": 0, "total_pages": 3},
        ),
        Document(
            page_content="Hermione Granger is a brilliant witch and Harry's best friend.",
            metadata={"source": "test_book.pdf", "page": 1, "total_pages": 3},
        ),
        Document(
            page_content="Ron Weasley is Harry's loyal friend from a pure-blood wizarding family.",
            metadata={"source": "test_book.pdf", "page": 2, "total_pages": 3},
        ),
    ]


@pytest.fixture
def sample_chunked_documents():
    """Create sample chunked Document objects for testing."""
    return [
        Document(
            page_content="Harry Potter is a young wizard",
            metadata={"source": "test_book.pdf", "page": 0, "start_index": 0},
        ),
        Document(
            page_content="who lives with his aunt and uncle.",
            metadata={"source": "test_book.pdf", "page": 0, "start_index": 30},
        ),
        Document(
            page_content="Hermione Granger is a brilliant witch",
            metadata={"source": "test_book.pdf", "page": 1, "start_index": 0},
        ),
    ]


class TestFileToLangchainDoc:
    """Test the file_to_langchain_doc function."""

    @pytest.mark.asyncio
    @patch("backend.RAG.PyPDF2.PdfReader")
    async def test_file_to_langchain_doc_success(
        self, mock_pdf_reader, sample_pdf_upload
    ):
        """Test successful conversion of PDF to Document objects."""
        # Setup mock
        pages_content = [
            "This is page 1 content",
            "This is page 2 content",
            "This is page 3 content",
        ]
        mock_pdf_reader.return_value = MockPyPDF2Reader(pages_content)

        # Call function
        result = await file_to_langchain_doc(sample_pdf_upload)

        # Assertions
        assert len(result) == 3
        assert all(isinstance(doc, Document) for doc in result)

        # Check first document
        assert result[0].page_content == "This is page 1 content"
        assert result[0].metadata["source"] == "test_book.pdf"
        assert result[0].metadata["page"] == 0
        assert result[0].metadata["total_pages"] == 3

        # Check that PDF was read
        sample_pdf_upload.read.assert_called_once()

    @pytest.mark.asyncio
    async def test_file_to_langchain_doc_empty_pdf(self):
        """Test handling of empty PDF."""
        mock_file = Mock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=b"")
        mock_file.filename = "empty.pdf"

        with patch("backend.RAG.PyPDF2.PdfReader") as mock_reader:
            mock_reader.return_value = MockPyPDF2Reader([])

            result = await file_to_langchain_doc(mock_file)
            assert len(result) == 0


class TestChunkLangchainPages:
    """Test the chunk_langchain_pages function."""

    def test_chunk_langchain_pages_default_params(self, sample_documents):
        """Test chunking with default parameters."""
        result = chunk_langchain_pages(sample_documents)

        # Should return Document objects
        assert all(isinstance(doc, Document) for doc in result)
        assert len(result) > 0

        # Check that start_index is added to metadata
        for doc in result:
            if "start_index" in doc.metadata:
                assert isinstance(doc.metadata["start_index"], int)

    def test_chunk_langchain_pages_custom_params(self, sample_documents):
        """Test chunking with custom parameters."""
        result = chunk_langchain_pages(
            sample_documents, chunk_size=50, chunk_overlap=10, add_start_index=False
        )

        assert all(isinstance(doc, Document) for doc in result)
        # With smaller chunk size, should get more chunks
        assert len(result) >= len(sample_documents)

    def test_chunk_langchain_pages_empty_input(self):
        """Test chunking with empty input."""
        result = chunk_langchain_pages([])
        assert result == []

    def test_chunk_langchain_pages_preserves_metadata(self, sample_documents):
        """Test that original metadata is preserved in chunks."""
        result = chunk_langchain_pages(sample_documents)

        for doc in result:
            # Should preserve original metadata
            assert "source" in doc.metadata
            assert "page" in doc.metadata or "total_pages" in doc.metadata


class TestEmbeddedPDF:
    """Test the EmbeddedPDF class."""

    @patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "test_token"})
    @patch("backend.RAG.HuggingFaceEndpointEmbeddings")
    def test_init(self, mock_embeddings):
        """Test EmbeddedPDF initialization."""
        mock_embeddings.return_value = MockHuggingFaceEmbeddings()

        pdf_embedder = EmbeddedPDF()

        assert pdf_embedder.db is None
        assert pdf_embedder.embedding_function is not None
        mock_embeddings.assert_called_once()

    @patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "test_token"})
    @patch("backend.RAG.HuggingFaceEndpointEmbeddings")
    @patch("backend.RAG.Chroma.from_documents")
    def test_embed_pdf_success(self, mock_chroma, mock_embeddings, sample_documents):
        """Test successful PDF embedding."""
        # Setup mocks
        mock_embeddings.return_value = MockHuggingFaceEmbeddings()
        mock_db = MockChroma(sample_documents)
        mock_chroma.return_value = mock_db

        pdf_embedder = EmbeddedPDF()
        result = pdf_embedder.embed_pdf(sample_documents)

        # Assertions
        assert result["success"] is True
        assert result["pages"] == len(sample_documents)
        assert "PDF processed successfully" in result["message"]
        assert pdf_embedder.db is not None
        mock_chroma.assert_called_once()

    @patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "test_token"})
    @patch("backend.RAG.HuggingFaceEndpointEmbeddings")
    @patch("backend.RAG.Chroma.from_documents")
    def test_embed_pdf_failure(self, mock_chroma, mock_embeddings, sample_documents):
        """Test PDF embedding failure."""
        # Setup mocks
        mock_embeddings.return_value = MockHuggingFaceEmbeddings()
        mock_chroma.side_effect = Exception("Database error")

        pdf_embedder = EmbeddedPDF()
        result = pdf_embedder.embed_pdf(sample_documents)

        # Assertions
        assert result["success"] is False
        assert "Database error" in result["error"]

    @patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "test_token"})
    @patch("backend.RAG.HuggingFaceEndpointEmbeddings")
    def test_semantic_search_no_database(self, mock_embeddings):
        """Test semantic search when no database is loaded."""
        mock_embeddings.return_value = MockHuggingFaceEmbeddings()

        pdf_embedder = EmbeddedPDF()

        with pytest.raises(ValueError, match="No PDF has been processed yet"):
            pdf_embedder.semantic_search("Harry Potter")

    @patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "test_token"})
    @patch("backend.RAG.HuggingFaceEndpointEmbeddings")
    @patch("backend.RAG.Chroma.from_documents")
    def test_semantic_search_success(
        self, mock_chroma, mock_embeddings, sample_documents
    ):
        """Test successful semantic search."""
        # Setup mocks
        mock_embeddings.return_value = MockHuggingFaceEmbeddings()
        mock_db = MockChroma(sample_documents)
        mock_chroma.return_value = mock_db

        pdf_embedder = EmbeddedPDF()
        pdf_embedder.embed_pdf(sample_documents)

        result = pdf_embedder.semantic_search("Harry Potter")

        # Should return formatted string with page content
        assert isinstance(result, str)
        assert "Harry Potter" in result or "[Page" in result

    @patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "test_token"})
    @patch("backend.RAG.HuggingFaceEndpointEmbeddings")
    @patch("backend.RAG.Chroma.from_documents")
    @patch("backend.RAG.InferenceClient")
    def test_generate_character_analysis(
        self, mock_client, mock_chroma, mock_embeddings, sample_documents
    ):
        """Test character analysis generation."""
        # Setup mocks
        mock_embeddings.return_value = MockHuggingFaceEmbeddings()
        mock_db = MockChroma(sample_documents)
        mock_chroma.return_value = mock_db
        mock_client.return_value = MockInferenceClient()

        pdf_embedder = EmbeddedPDF()
        pdf_embedder.embed_pdf(sample_documents)

        result = pdf_embedder.generate_character_analysis("Harry Potter")

        # Should return string analysis
        assert isinstance(result, str)
        assert len(result) > 0
        mock_client.assert_called_once()

    @patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "test_token"})
    @patch("backend.RAG.HuggingFaceEndpointEmbeddings")
    def test_has_documents_false(self, mock_embeddings):
        """Test has_documents when no database is loaded."""
        mock_embeddings.return_value = MockHuggingFaceEmbeddings()

        pdf_embedder = EmbeddedPDF()
        assert pdf_embedder.has_documents() is False

    @patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "test_token"})
    @patch("backend.RAG.HuggingFaceEndpointEmbeddings")
    @patch("backend.RAG.Chroma.from_documents")
    def test_has_documents_true(self, mock_chroma, mock_embeddings, sample_documents):
        """Test has_documents when database is loaded."""
        # Setup mocks
        mock_embeddings.return_value = MockHuggingFaceEmbeddings()
        mock_db = MockChroma(sample_documents)
        mock_chroma.return_value = mock_db

        pdf_embedder = EmbeddedPDF()
        pdf_embedder.embed_pdf(sample_documents)

        assert pdf_embedder.has_documents() is True


class TestIntegration:
    """Integration tests for the RAG system."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "test_token"})
    @patch("backend.RAG.PyPDF2.PdfReader")
    @patch("backend.RAG.HuggingFaceEndpointEmbeddings")
    @patch("backend.RAG.Chroma.from_documents")
    @patch("backend.RAG.InferenceClient")
    async def test_full_workflow(
        self,
        mock_client,
        mock_chroma,
        mock_embeddings,
        mock_pdf_reader,
        sample_pdf_upload,
    ):
        """Test the complete workflow from PDF upload to character analysis."""
        # Setup mocks
        pages_content = [
            "Harry Potter is a young wizard who lives with his relatives.",
            "Hermione Granger is Harry's brilliant friend who helps him.",
        ]
        mock_pdf_reader.return_value = MockPyPDF2Reader(pages_content)
        mock_embeddings.return_value = MockHuggingFaceEmbeddings()

        # Create sample documents that would be generated
        documents = [
            Document(
                page_content=content,
                metadata={"source": "test_book.pdf", "page": i, "total_pages": 2},
            )
            for i, content in enumerate(pages_content)
        ]

        mock_db = MockChroma(documents)
        mock_chroma.return_value = mock_db
        mock_client.return_value = MockInferenceClient()

        # Run the workflow
        pdf_embedder = EmbeddedPDF()

        # Step 1: Convert PDF to documents
        documents = await file_to_langchain_doc(sample_pdf_upload)
        assert len(documents) == 2

        # Step 2: Embed the PDF
        embed_result = pdf_embedder.embed_pdf(documents)
        assert embed_result["success"] is True

        # Step 3: Search for character
        search_result = pdf_embedder.semantic_search("Harry Potter")
        assert isinstance(search_result, str)

        # Step 4: Generate analysis
        analysis = pdf_embedder.generate_character_analysis("Harry Potter")
        assert isinstance(analysis, str)
        assert len(analysis) > 0
