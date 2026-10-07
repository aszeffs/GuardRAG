from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from guardrag.answer import AskRequest, AskResponse
from guardrag.config import Settings, get_settings
from guardrag.db import connect
from guardrag.embedder import FastEmbedEmbedder
from guardrag.grounding import Grounder, ground, out_of_scope
from guardrag.guards import InjectionClassifier, PromptGuard, RateLimiter, screen_for_injection
from guardrag.llm import LLM, GroqLLM
from guardrag.pii import asks_for_personal_data, install_log_redaction, redact
from guardrag.retrieval import Retriever, build_retriever


def create_app(
    *,
    retriever: Retriever,
    llm: LLM,
    classifier: InjectionClassifier,
    grounder: Grounder = ground,
    rate_limiter: RateLimiter | None = None,
    k: int = 5,
) -> FastAPI:
    """The GuardRAG API, built from its dependencies so tests can pass in fakes.

    Without a `rate_limiter`, requests are not rate limited. Building the app turns on log
    redaction for the whole process (#11).
    """
    install_log_redaction()
    app = FastAPI(title="GuardRAG")
    app.state.retriever = retriever
    app.state.llm = llm
    app.state.classifier = classifier
    app.state.rate_limiter = rate_limiter

    def within_rate_limit(request: Request) -> None:
        # Runs before the body is validated, so malformed requests count towards the limit too.
        client = request.client.host if request.client else "unknown"
        if rate_limiter is not None and not rate_limiter.allow(client):
            retry_after = str(round(rate_limiter.window_seconds))
            raise HTTPException(429, "Too many requests", headers={"Retry-After": retry_after})

    @app.exception_handler(RequestValidationError)
    async def validation_error_without_input(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI's default 422 echoes the rejected question, which may hold personal data.
        errors = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]
        return JSONResponse({"detail": jsonable_encoder(errors)}, status_code=422)

    @app.post("/ask", dependencies=[Depends(within_rate_limit)])
    def ask(request: AskRequest) -> AskResponse:
        question = request.question
        if asks_for_personal_data(question) or screen_for_injection(question, classifier):
            return out_of_scope()
        passages = retriever.search(question, k)
        response = grounder(llm.draft(question, passages), passages)
        # Contact details the Passages give (an Agency's hotline) may be repeated; nothing else.
        answer = redact(response.answer, keep=[p.text for p in passages])
        return response.model_copy(update={"answer": answer})

    @app.get("/health")
    def health() -> dict[str, object]:
        with connect() as conn:
            passages = conn.execute("SELECT count(*) FROM passages").fetchone()[0]
        return {"status": "ok", "passages": passages}

    return app


def create_production_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    return create_app(
        retriever=build_retriever(settings.retriever, embedder),
        llm=GroqLLM(settings.groq_api_key, settings.answer_model),
        classifier=PromptGuard(settings.groq_api_key, settings.prompt_guard_model),
        rate_limiter=RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds),
        k=settings.retrieval_k,
    )


app = create_production_app()
