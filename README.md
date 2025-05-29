## CharMem: Never lose track of those minor characters again!

If you've taken a break from a book or series and can't remember who a minor character is, CharMem is here to help. This tool allows you to search for characters by name and provides a brief description of each character and recent events they were involved in.

## Features

-   Semantic search for characters by name (only looking up to where the user has read)
-   Brief descriptions of characters
-   Recent events involving characters
-   Response in a way that matches the story e.g. in the way that the main character or narrator would describe them (would this involve fine-tuning?)

## Pipeline

1. Query
2. Embed query
3. Chunk the corups and get top-k similar chunks
4. Retrieve context + query
5. Feed in to LLM
6. Generate answer
