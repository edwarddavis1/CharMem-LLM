## CharMem: Never lose track of those minor characters again!

If you've taken a break from a book or series and can't remember who a minor character is, CharMem is here to help. This tool allows you to search for characters by name and provides a brief description of each character and recent events they were involved in.

## Features

-   Semantic search for characters by name (only looking up to where the user has read)
-   Brief descriptions of characters
-   Recent events involving characters
-   Response in a way that matches the story e.g. in the way that the main character or narrator would describe them

## Currently Implemented

-   **PDF Upload**: Users can upload PDF books through a web interface
-   **RAG-powered Character Search**: Semantic search through uploaded PDFs using vector embeddings
-   **Real-time Chat Interface**: WebSocket-based chat for instant responses
-   **LLM Integration**: Uses Hugging Face Inference API with Qwen 235B model
-   **Character Analysis**: Provides character summaries, first meeting details, and recent events with page references
-   **Vector Database**: Chroma DB for document chunking and similarity search (k=50 chunks)
-   **Responsive UI**: Modern dark-themed chat interface with typing indicators
