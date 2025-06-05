# About this chatbot project

## Backend

In the backend I've used python and FastAPI.

**Python** was required to use the huggingface API, allowing for the catelogue of language models to be run on the hugging face cloud infrastructure. This is important because running these models locally is not feasible due to their size and resource requirements.

**FastAPI** was used to implement websockets which allows the backend and frontend to communicate in a two way, real-time system. This was preferred over a REST API, (one way, request-response), as websockets allows for instant delivery of messages (rather than checking every 2s for example) and is more efficient.

### Backend Details

**Logging**: for debugging and troubleshooting

-   Logging was used as it is useful for aiding the debugging process - much better than just using a load of random print statements!
-   The logging for the app tracks _startup_, _HF API interactions_, _error handling_ and _response validation_.
-   E.g. whenever regular python code handles an exception, the error message is given to the logger via `logger.error(error_msg)`.

**`load_dotenv()`**: for API key security

-   Handles secret variables - i.e. hugging face API key. This is stored in a `.env.` file, which is included in the `.gitignore`. This makes sure that the API key is never accidentally pushed to the public repo.

**Jinja2**: template rendering

Jinja is a python library that allows for dynamic HTML pages. It essientally allows you to have variables in the HTML code. For example, if I want to dynamicically display the language model in use for the chatbot, I can place `{{ model_name }}` in the HTML, and then use Jinja in python to define this model name. This is done using

```python
templates.TemplateResponse("index.html", {"model_name": MODEL_ID})
```

where `templates = Jinja2Templates(...)`. So this `TemplateResponse()` function scans through the HTML and updates the variables in the raw HTML.

**WebSocket `ConnectionManager`**: message handling

Used to track active connections, handle connecting/disconnecting from a server and sendings messages to the server. Note that in the cases where python is sending or recieving data from the websocket, we need to use `async` functions.

```python
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
```

**Hugging Face**: LLM handling

Hugging face is similar to GitHub, except it stores and handles AI models, e.g. language models, image models, ...

In order to access a language model for our chat (e.g. Deepseek V3, 671B param), we need a hugging face API key. Once this is obtained, we can instantiate the `InferenceClient` by plugging in the API key. Once this is done, we can perform a query with a model using

```python
completion = client.chat.completions.create(
    model=MODEL_ID,
    messages=conversation_history,
    max_tokens=200,
    temperature=0.7,
)
bot_response = completion.choices[0].message.content
```

where `conversation_history` is the entire conversation (both user inputs and bot replies) up to this point. Each entry in the conversation history has the format `{"role":role, "content":content}`, where the role is e.g. user or assistant, and content is the message.

**Endpoints**

There are two endpoints in this project:

1. **HTTP endpoint**: This is a GET request which grabs the template `index.html` (on which Jinja does template rendering).
2. **WebSocket endpoint**: Handles the chat back-and-forth; both recieving and sending messages.

**Uvicorn**: run the backend

Run the FastAPI app using

```python
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

Use this over just `python main.py` or `uv run main.py` as uvicorn allows for updates to the FastAPI app on save. In the latter cases the app would have to be restarted for changes to take effect.
