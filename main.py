import os
from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    UploadFile,
    File,
)
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import logging
from pathlib import Path
from huggingface_hub import InferenceClient
from backend.RAG import EmbeddedPDF, file_to_langchain_doc
from backend.config import MODEL_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Initialize the InferenceClient for the new Inference Providers system
client = InferenceClient(api_key=HF_API_TOKEN) if HF_API_TOKEN else None

if not HF_API_TOKEN:
    logger.error("HUGGINGFACE_API_TOKEN not found in environment variables!")
else:
    logger.info(f"Initialized InferenceClient with model: {MODEL_ID}")


app = FastAPI(title="ChatBot App with Hugging Face LLM")

app.state.pdf_embedder = None

# Mount static files
app.mount("/static", StaticFiles(directory=Path("app/static")), name="static")

# Templates
templates = Jinja2Templates(directory=Path("app/templates"))


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


manager = ConnectionManager()


async def query_huggingface(conversation_history):
    """
    Query the Hugging Face Inference Providers API with the given message
    Uses the modern InferenceClient with chat completion format
    """
    try:
        if not client:
            return {
                "error": "InferenceClient not initialized. Check your HUGGINGFACE_API_TOKEN."
            }

        logger.info(f"Sending request to Hugging Face Inference Providers")
        logger.info(f"Model: {MODEL_ID}")
        logger.info(f"Conversation length: {len(conversation_history)}")

        # Use the new chat completion format
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=conversation_history,
            max_tokens=1000,
            temperature=0.7,
        )
        # Extract the response
        bot_response = completion.choices[0].message.content

        if bot_response:
            logger.info(f"Received successful response: {bot_response[:100]}...")
            return {"response": bot_response}
        else:
            error_msg = "Received empty response from the model"
            logger.warning(error_msg)
            return {"error": error_msg}

    except Exception as e:
        error_msg = f"Exception occurred: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "app_name": "CharMem", "model_name": MODEL_ID},
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        app.state.conversation_history = []

        while True:
            data = await websocket.receive_text()
            user_message = data.strip()

            # If available use the uploaded PDF as context for the user message
            # A semantic search through the PDF will return relevant excerpts
            if app.state.pdf_embedder is not None:
                retrieval = app.state.pdf_embedder.semantic_search(user_message)

                app.state.conversation_history.append(
                    {
                        "role": "system",
                        "content": f"The following information may or may not be relevant to the following user query. If you think that this information is relevant, reference it and give page numbers: {retrieval}",
                    }
                )

            # Add user message to conversation history (using ChatML formatting)
            app.state.conversation_history.append(
                {"role": "user", "content": user_message}
            )

            await manager.send_message("Bot is thinking...", websocket)
            response = await query_huggingface(app.state.conversation_history)

            if "error" in response:
                bot_reply = f"Error: {response['error']}"
                logger.error(f"Error from Hugging Face: {response['error']}")
            elif "response" in response and response["response"]:
                bot_reply = response["response"].strip()
            else:
                bot_reply = "Sorry, I couldn't understand the model's response."

            # Add bot reply to conversation history (using ChatML formatting)
            app.state.conversation_history.append(
                {"role": "assistant", "content": bot_reply}
            )

            # Send bot's reply to the client
            await manager.send_message(bot_reply, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await manager.send_message(f"Connection error: {str(e)}", websocket)
        except:
            pass
        manager.disconnect(websocket)


@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    # Validate file type
    if not pdf.filename or not pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pages = await file_to_langchain_doc(pdf)

    app.state.pdf_embedder = EmbeddedPDF()
    result = app.state.pdf_embedder.embed_pdf(pages)

    if result["success"]:
        return {
            "message": result["message"],
            "pages": result["pages"],
        }
    else:
        raise HTTPException(
            status_code=500, detail=f"Error processing PDF: {result['error']}"
        )


@app.post("/query-character")
async def query_character(character_name: str):
    """Query character information from uploaded PDFs."""
    try:
        if app.state.pdf_embedder is None:
            raise HTTPException(
                status_code=400,
                detail="No PDF has been uploaded yet. Please upload a PDF first.",
            )

        analysis = app.state.pdf_embedder.generate_character_analysis(character_name)
        return {"character": character_name, "analysis": analysis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
