from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai.errors import APIError
from pydantic import ValidationError

from ai_playground.llm import (
    EmptyPromptError,
    EmptyResponseError,
    create_client,
    get_structured_async,
)
from ai_playground.schemas import MeetingNotes, Summary, TextRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = create_client()
    yield
    app.state.client.close()


app = FastAPI(lifespan=lifespan)


def get_client(request: Request) -> genai.Client:
    return request.app.state.client


Client = Annotated[genai.Client, Depends(get_client)]


@app.exception_handler(EmptyPromptError)
async def empty_prompt_handler(request: Request, exc: EmptyPromptError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(EmptyResponseError)
async def empty_response_handler(request: Request, exc: EmptyResponseError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(ValidationError)
async def schema_mismatch_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=502,
        content={"detail": "Gemini's response didn't match the expected format."},
    )


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    if exc.code == 429:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limited by the AI provider. Try again shortly."},
        )
    return JSONResponse(
        status_code=502,
        content={"detail": "The AI provider returned an error."},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=Summary)
async def summarize(body: TextRequest, client: Client) -> Summary:
    return await get_structured_async(client, body.text, Summary)


@app.post("/meeting-notes", response_model=MeetingNotes)
async def meeting_notes(body: TextRequest, client: Client) -> MeetingNotes:
    return await get_structured_async(client, body.text, MeetingNotes)
