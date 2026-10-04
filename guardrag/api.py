from fastapi import FastAPI

from guardrag.answer import AskRequest, AskResponse
from guardrag.config import Settings, get_settings
from guardrag.db import connect
from guardrag.embedder import FastEmbedEmbedder
from guardrag.grounding import Grounder, ground
from guardrag.guards import InjectionClassifier, NoClassifier
from guardrag.llm import LLM, GroqLLM
from guardrag.retrieval import Retriever, build_retriever


def create_app(
    *,
    retriever: Retriever,
    llm: LLM,
    classifier: InjectionClassifier,
    grounder: Grounder = ground,
    k: int = 5,
) -> FastAPI:
    """The GuardRAG API, built from its dependencies so tests can pass in fakes."""
    app = FastAPI(title="GuardRAG")
    app.state.retriever = retriever
    app.state.llm = llm
    app.state.classifier = classifier  # screening before retrieval lands with #10

    @app.post("/ask")
    def ask(request: AskRequest) -> AskResponse:
        passages = retriever.search(request.question, k)
        draft = llm.draft(request.question, passages)
        return grounder(draft, passages)

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
        classifier=NoClassifier(),
        k=settings.retrieval_k,
    )


app = create_production_app()
