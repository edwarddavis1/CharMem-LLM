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
4. [ ] Perform RAG on the uploaded data

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

-   [ ] Embed the file in chunks (if the file has not been seen before)
-   [ ] Perform a semantic search on the chunks
-   [ ] Engineer a prompt to the model including the retrieval as context.
-   [ ] Call the hugging face client with the prompt

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
