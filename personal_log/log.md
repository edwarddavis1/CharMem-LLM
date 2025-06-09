# **2025-5-28 Wed**

Project start.

Want a project which gives me experience building a product that uses an LLM and RAG.

The idea: utilise semantic search to provide a character summary about specific characters in a book, only including text up to the point where the user has read.

## Retrieval-Augmented Generation (RAG)

### High Level

High-level idea: when queried, the LLM retreives some relevant content before answering.
Here, "relevant content" could mean looking up on the internet, or looking through a pre-determined database. In the case of this project, that will be locating information in the pdf regarding the queried character.

#### Benefits of RAG

-   Reduces hallucination: given that the model has data to first query, if the question cannot reliably be answered based on this data, it allows the model to say "I don't know".
-   Answers are more up to date.

#### Active research directions

-   How to make a good retriever of relevant information.
-   How to best generate an answer given the retrieved data.

### More detail

The overall pipeline looks like the following.

1. Query
2. Embed query
3. Chunk the corups and get top-k similar chunks
4. Retrieve context + query
5. Feed in to LLM
6. Generate answer

# **2025-5-29 Thu**

# Getting the initial results

## Prompt engineering

I wanted the output to come in three parts, given a query about a specific character.

1. A summary of the character.
2. Where we first met the character (inc page number + how they were introduced)
3. Some recent events.

I'm getting this using a prompt template

```{python}
PROMPT_TEMPLATE = """
You are a helpful book assistant....

Context: {retrieval}

Question: Who is {query}? Where did we first meet ....

Answer in the following style

Example 1:
...


Answer:
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=retrieval, query=query_text)
```

## Language model

Now using google colab, allowing me to run the code using a GPU.

-   Started with a pretty small model - "tiiuae/falcon-rw-1b", results weren't amazing.
-   Switched to a bigger model - "mistralai/Mistral-7B-Instruct-v0.3" - results are better.

Currently hyperparameters like temperature and top_p have been kept to defaults.

## Chunking

One initial problem was the output incorrectly placing where we first met a character. This was due to the chunk where we meet said character not being included in the retrieval.

Simply increasing the number of chunks returned by similarity search was too computationally taxing.

By reducing the chunk size to around 200 (instead of 1000), this first meet result was improved. However, a tradeoff could be the quality of character summary, as the model will only see a small amount of text for each chunk.

## Example output: Hermione Granger

-   Correct first meet location.
-   Accurate summary, although a little shallow
-   First recent event is correct, the second is incorrect.

```
A summary of the character Hermione Granger is she is a character who is knowledgeable and independent, as seen on page 77 when she introduces herself to Harry.

We first met Hermione Granger on page 77, this was when she introduced herself to Harry Potter.

Recent important events involving Hermione Granger include her attempt to deal with a troll on her own on page 129, and her not speaking to Ron or Harry since the day Harry’s broomstick arrived on page 124.
```

Not bad, but lots of room for improvement.

### Reduce chunk size to 100

The output is hilariously vauge.

Hermione Granger is a character in the novel.

We first met Hermione Granger on page 77 [correct].

Recent important events involving Hermione Granger include her being present with Ron on page 124, standing up on page 129, and being mentioned on page 182.

# **2025-5-30 Fri**

## Progress: Looking at CUDA

Colab has been working well, however, there are time limits when using GPUs on the free tier.

Therefore, I've moved over to Windows and set up CUDA to use my local GPU (Nvidia GTX 960, 4GB Mem).

This is able to run small models (~1B param), but it is too small to run the Mistral one that I was using before on colab (~7B param).

I am able to run "microsoft/Phi-4-mini-instruct" locally (~4B param) - but it is very slow to run (~6min to instantiate, ~3min to tell me about Hermione).

## Short term strategy

Stop trying to use larger LLMs - most of the room for improvement comes from the retrieval part of the code. All I need the LLM to do is summarise the retrieval - which hopefully doesn't require a huge model.

## How would I actually deploy this?

| Stack Layer   | Tech                   | Role                             |
| ------------- | ---------------------- | -------------------------------- |
| Frontend      | HTML/JS/Typescript     | User input & result display      |
| API Backend   | FastAPI (Python)       | Wraps LangChain + Chroma logic   |
| Data Pipeline | LangChain              | Handles PDF → vector → LLM query |
| Hosting       | GitHub Pages + Railway | Free deployment                  |

-   This might end up being a distrction - but I can just keep writing python code, and if I decide I want to delopy it as an MVP, then I can use this plan without altering my proof of concept code.

## Examples with local LLMs

Note that each of these examples are based on a pretty bad retrieval model at the moment.

There may well be an issue of the model referring to its internal knowledge, instead of what information is given to them in the retrieval. This could cause accidental spoilers...

### tiiuae/falcon-rw-1b (4.88 GB)

_Time to instantiate_: 5s
_Time to generate_: 33s

Character: Who is Hermione Granger?

Answer:
She is a student at Hogwarts School of Witchcraft and Wizardry. **She is the daughter of Albus Dumbledore and Molly Weasley.** Hermione is a Gryffindor, and is currently a first year student at Hogwarts. She has a reputation as being very smart, and is popular with her friends.

Verdict:

-   Not very accurate hahaha
-   Does not follow the formatting instructions in my prompt template
-   Does not reference page numbers

### LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct (8.96 GB)

_Time to instantiate_: 35s
_Time to generate_: 3m 44s

Answer:
Hermione Granger is a prominent and intelligent young witch who plays a crucial role in the Harry Potter series. She is known for her exceptional academic abilities, loyalty to her friends, and a strong sense of justice, often acting against authority when necessary to protect others. Hermione is fiercely independent, quick-witted, and possesses extensive knowledge of magical lore and spells. She is part of the Gryffindor house at Hogwarts School of Witchcraft and Wizardry alongside Harry Potter and Ron Weasley.

**We first meet Hermione Granger on page XX**, where she is introduced as Hermione Granger, alongside Ron Weasley and Harry Potter, in a crowded classroom setting after they have inadvertently caused a commotion by attempting to retrieve a stolen book from the Restricted Section. They are recognized by Hagrid, who quickly identifies them as students of Hogwarts, leading to Hermione sharing her extensive magical knowledge, particularly regarding centaurs, which marks her introduction to the community.

Recently, Hermione Granger has been involved in various significant events: **On page 182**, she encounters a centaur named Ronan at the school, leading to an interesting exchange about astronomy and magical creatures. Additionally, Hermione faces disciplinary action from Professor McGonagall for her actions in confronting a mountain troll alone, highlighting her willingness to defy rules when necessary for the greater good.

Verdict:

-   Much more significant responses
-   Follows the formatting instructions
-   _Can_ reference page numbers, but in the second paragraph it overfits to the prompt - literally returning "page XX". Perhaps this could be fixed with better prompt engineering

### microsoft/Phi-4-mini-instruct (7.16 GB)

_Time to instantiate_: 38s
_Time to generate_: 2m 53s

Answer: Hermione Granger is a character from the Harry Potter series, known for her intelligence, bravery, and loyalty. She is a dedicated student at Hogwarts School of Witchcraft and Wizardry, where she excels academically and is a member of the Gryffindor house. Hermione is also known for her strong sense of justice and her willingness to stand up for what she believes in, often putting herself in danger to help others.

**We first meet Hermione Granger on page 130**, where she is introduced as a student at Hogwarts. She is standing alone by the door, waiting for Harry, Ron, and Neville Longbottom, who have just escaped from the trap set by the Fat Lady, the portrait of the Gryffindor House's mascot. Hermione is described as being alone, standing by the door, waiting for them, and she is portrayed as a quiet and reserved character at this point in the story.

Recently, **on page 182**, Hermione Granger is seen interacting with Ronan, a centaur, who introduces himself as a student at Hogwarts. Ronan asks Hermione if she learns much at the school, to which she responds timidly that she does a bit. This interaction shows Hermione's willingness to engage in conversation with others and her continued dedication to her studies.

Verdict:

-   Gave full answers
-   Used the correct format
-   Coud reference pages

## Result

The Microsoft Phi4 model gave the best answers (given the poor retrieval context) and ran quicker than the other model of a similar size. Due to the terrible quality of the tiiuae 1B model; ignoring formatting instructions, being very breif and suffering from hallucination, I think it would be best to focus on the Phi4 model for now.

While this can run locally fine, having to wait 2-3 minutes per run is annoying - so perhaps I'll do most of the LLM experimentation on google colab, and do the rest of the development using VS code (e.g. improving retrieval performance, front end etc..)

# **2025-5-31 Sat**

## First course of action to improve retrieval

### Promot engineering

-   First attempt at prompt engineering might be a bit random and bloated. Could try a different technique using ### Role, ### Constraints.

### Standard LLM params

-   **Temperature** is used to control the randomness of the model's output. Lower values (e.g., 0.1) make the output more deterministic and focused, while higher values (e.g., 0.8) increase creativity and diversity in responses.
-   **Top_p (nucleus sampling)** limits the model to considering only the most probable tokens whose cumulative probability exceeds the threshold p. This helps balance diversity and coherence.

# **2025-6-1 Sun**

## Key Issue: LLM Generation Latency

So far I've shown that it's possible to make a minimum viable product using a LLM of around 4B parameters. However, running locally, there is a generation latency on the scale of minutes - which is clearly not good enough for an MVP.

### Possible latency solutions

#### Backend

-   FastAPI: Web framework for building API apps with Python.

    > Easy to learn
    > Fast development
    > High performance

-   AWS:

#### LLM

-   While Phi-4 seemed okay, it could even be worth testing out Phi-2.

#### Vector database

-   FAISS is generally faster than Chroma (although Chroma is better-integrated with LangChain).

# [2025-6-5 Thu]

## Key Issue Solved: Hugging face inference client

Spent a while away from this project to learn about FastAPI and adjacent tools. Here is a summary of [how I created a huggingface chatbot webpage](making_the_chatbot.md).

But importantly, huggingface inference client has solved my LLM latency problem, and it also allows us to use the most powerful models, e.g. deepseek (>650B param!).

# [2025-6-6 Fri]

## Getting to an MVP

The biggest gaps to the minimum viable product is the frontend. This needs the following to be an MVP.

1. [x] Allow for the user to upload a pdf
2. [ ] Allow the reader to scroll through the pdf (to read it)
3. [ ] Take note of the page that the user has reached.
4. [x] Perform RAG on the uploaded data

## Uploading the PDF

-   Add an event listener for the button in Javascript.
-   When the `pdf-upload` input HTML element is clicked, the file browser opens

    HTML: `<input type="file" id="pdf-upload" accept=".pdf" style="display: none;"/>`

    Javascript: `pdfUpload.click();`

-   When the file is selected run `uploadPDF(file)`. The first part of this function uses a `FormData` object to package the selected file, allowing it to be sent via a POST HTTP request to `/upload-pdf`.

-   Then need to add an endpoint at `/upload-pdf` in the FastAPI app. The python function `upload_pdf(pdf)` reads the PDF using `PyPDF2` and sends a response back to Javascript with either an error (file type error, or unknown error) or with success. On a successful interaction, the return object is (currently) a logging `message`, `pages` count and `text_preview`.

-   The return from the python function is then collected by the uploadPDF js function: `const result = await response.json()`. The remainder of this function then handles error or success messages.

## Adding RAG to the chatbot

In the above implementation of the file upload, python simply reads the file and returns a success message without actually doing anything with the data. To perform RAG on the PDF we need the following:

-   [x] Embed the file in chunks (if the file has not been seen before)
-   [x] Perform a semantic search on the chunks
-   [x] Engineer a prompt to the model including the retrieval as context.
-   [x] Call the hugging face client with the prompt

### Challenge: speeding up document embedding

The embedding of all of the chunks of the pdf takes time. On the Harry Potter book 1 example the chunking of the pdf is very quick. However, loading the embedding function (all-MimiLM-L6-v2) takes 1m30s! Once this is loaded, the VB computation is around 10s - not the quickest but definitely a manageable delay.

#### Speedup options

-   Use a different VB management system - e.g. Faiss
-   Use a lighter embedding model
-   Use the HF inference client or something similar to offload compute
-   Cache the embedding function - i.e. start loading as the page loads. This cuts down the percieved loading time and removes the loading time altogether when doing a second request.

#### Speedup Progress

**Embedding function instantiation**: 1m30s -> 0.0s

-   Use the hugging face inference client, just like what was done with the LLM computation.
-   Old:

    ```python
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    ```

-   New

    ```python
    embedding_function = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_ID, huggingfacehub_api_token=HF_API_TOKEN
    )
    ```

-   Because this speedup was so successful, there is no need to implement the embeding function caching idea.

-   As the VDB computation with Chroma is not too slow (~10s), I'll stop trying to speed this up for now. Partly due to having to configure FAISS to use GPU, which I'll just have in my mind as a lower-priority job.

### Overall computation time of RAG for Harry Potter book 1

(Ignoring computations that round to 0.0s)

-   **Load the PDF**: 2.1s
-   **Compute the VDB (Chroma)**: 10.3s
-   **Semantic search (k=3)**: 0.1s
-   **LLM response (Qwen 235B)**: 13.8s

-   **_Total_**: 26.3s

Using a smaller model

-   **LLM response (Phi-4 14.7B)**: 3.0s

-   **_Total_**: 15.5s

## RAG Performance Improvements

-   The RAG process improved a lot by increasing the value of `k` in the semantic search. This was capped at `k=3` in the original google colab or locally run versions of this project. Now it's possible to increase it to much larger values. I settled on around `k=50` for now.
-   This allows for the model to incorporate more events into its context, which also increases the change of including the correct meeting point - something that has been difficult up to now.

-   While Qwen (235B) is a lot slower than Phi-4 (14.7B), it is noticably better. Phi-4 can give wrong info and is not able to say "character not met" like instructed. A model will most likely exist that does as well as Qwen, but is a lot smaller (perhaps a distilled one). The task doesn't seem to be that hard so there isn't a large emphasis on having the largest model.
-   The process to find such a model, however, is unclear.

## End of day summary and next steps

-   RAG is now implemented in the app.
-   You can upload a pdf and then every subsequent message to the chatbot will be used as a semantic search term through the PDF, then the chatbot will response based on that context.

### Current issues

-   [ ] The way I've implemented the system means that after every message (after PDF upload) a semantic search, returning `k=50` 1,000-character chunks of text is computed and added to the context. While the computation of this 50 chunks is actually pretty quick (~0.1s) the large context is most likely slowing generation.
-   [x] No tests in place yet!

### RAG improvements

-   Now is the time where it's a good idea to go back to literature. I have everything in place to start experimenting with different RAG paramters and different language models.
-   It may even be worth setting up benchmarks for different RAG + LLM parameter changes, e.g. accuracy of first meeting (requires manually labelling), accuracy of not meeting the character (easy to get a lot of data after the aforementioned labelling), and accuracy of information provided (unclear how to define a metric).

# [2025-6-7 Sat]

## Testing Python RAG functions

### Mock Testing

Mock testing can be used to replace API calls.
The syntax for adding mocks in place of API calls is as follows:

```python
@patch("A")  ← Matches first parameter
@patch("B")  ← Matches second parameter
@patch("C")  ← Matches third parameter
def test(self, param1, param2, param3):
           ↑       ↑       ↑
           A       B       C
```

Let's look at an example in the codebase. The current form of the `generate_character_analysis` function uses `Chroma`, `HuggingFaceEndpointEmbeddings`, and `InferenceClient` to perform the RAG process. We can mock these to test the function without making actual API calls. The order of the `@patch` decorators and the arguments of the test function are used to assign mock classes in place of the real, expensive classes.

```python
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
```

With this in place, we simply define "mock" classes which are just classes with the same structure as the real classes with some made-up data. For example, the `MockInferenceClient` class can be defined as follows:

```python
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

    def create(self, messages=None):
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
```

As this _mocks_ the `response = client.chat.completions.create()` structure used in `generate_character_analysis`.

### Coverage Testing

Coverage testing is a way to ensure that all parts of the code are tested. In Python, this can be done using the `coverage` package. It tracks which lines of code are executed during tests and reports on the coverage percentage.

When using pytest, you can run

```bash
pytest --cov=backend
```

to run the tests and check the coverage of the backend directory.

To make this even easier, I've added two lines to the `pyproject.toml` which runs the coverage check automatically when running `pytest`:

```toml
[tool.pytest.ini_options]
addopts = "--cov=backend"
```

Example:

```bash
Name                Stmts   Miss  Cover
---------------------------------------
backend\RAG.py         57      0   100%
backend\config.py       2      0   100%
---------------------------------------
TOTAL                  59      0   100%
```

### Continuous Integration

#### Workflows

To make the code more robust, I've added the `test-and-lint.yaml` GitHub Actions workflow to build the project and run the tests automatically (with coverage) on every push and pull request to `main` and to lint the code using _ruff_.

#### Pre-commits

One thing I haven't come across before is the idea of pre-commits. These are scripts that run before a commit is made, allowing you to check the code for errors or formatting issues before it is committed to the repository.

This is great because if you accidentally try to push before running tests or linting, it will fail and you can fix the issues without having to wait for the CI to run.

The instructions for the pre-commit are in the `.pre-commit-config.yaml` file, and the `pre-commit` package is requires - I've put this under the `dev` dependencies in the `pyproject.toml`.

Example:

```bash
> git commit -m "Add new feature"
check yaml...............................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
check for added large files..............................................Passed
check for merge conflicts................................................Passed
ruff (legacy alias)..................................(no files to check)Skipped
ruff format..........................................(no files to check)Skipped
pytest-check.............................................................Passed
```

# [2025-6-9 Mon]

## Next steps

-   [ ] Docker
-   [ ] Look into agentic RAG and _Reasoning Based Retrieval_.
-   [ ] Think about how to fix the hallucination issues.

### Hallucination issues

In the context of this project, hallucination refers to the LLM generating information that is not present in the PDF (regardless of whether it is correct or not).

For example, when asked about Hermione Granger, CharMem introduced her as Hermione _Jean_ Granger. This middle name is not present in the PDF, meaning that the model was using information it learned outside of the PDF. This is not ideal behaviour.
