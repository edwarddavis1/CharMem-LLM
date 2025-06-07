"""
Test utilities and fixtures for RAG tests.
"""


class MockPyPDF2Reader:
    """Mock PyPDF2.PdfReader for testing."""

    def __init__(self, pages_content):
        self.pages = [MockPage(content) for content in pages_content]


class MockPage:
    """Mock PDF page for testing."""

    def __init__(self, content):
        self.content = content

    def extract_text(self):
        return self.content


class MockChroma:
    """Mock Chroma vector database for testing."""

    def __init__(self, documents=None):
        self.documents = documents or []
        self.embedding_function = None

    @classmethod
    def from_documents(cls, documents, embedding_function):
        instance = cls(documents)
        instance.embedding_function = embedding_function
        return instance

    def similarity_search_with_relevance_scores(self, query, k=10):
        # Return mock search results
        return [(doc, 0.9) for doc in self.documents[:k]]


class MockHuggingFaceEmbeddings:
    """Mock HuggingFace embeddings for testing."""

    def __init__(self, model=None, huggingfacehub_api_token=None):
        self.model = model
        self.api_token = huggingfacehub_api_token

    def embed_documents(self, texts):
        # Return mock embeddings
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class MockInferenceClient:
    """Mock Hugging Face InferenceClient for testing."""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat = MockChat()


class MockChat:
    """Mock chat object for testing."""

    def __init__(self):
        self.completions = MockChatCompletions()


class MockChatCompletions:
    """Mock chat completions for testing."""

    def create(self, model=None, messages=None, temperature=None):
        # Return a mock response based on the input
        user_message = messages[0]["content"] if messages else ""

        if "Harry Potter" in user_message:
            content = "Harry Potter is the main protagonist of the series."
        elif "Hermione" in user_message:
            content = (
                "Hermione Granger is a brilliant witch and one of Harry's best friends."
            )
        else:
            content = "Character information not found in the provided context."

        return MockChatResponse(content)


class MockChatResponse:
    """Mock chat response for testing."""

    def __init__(self, content):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock choice in chat response for testing."""

    def __init__(self, content):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message in choice for testing."""

    def __init__(self, content):
        self.content = content
