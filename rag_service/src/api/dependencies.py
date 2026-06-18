from functools import lru_cache

from fastapi import Request


@lru_cache(maxsize=1)
def get_rag_pipeline(request: Request) -> object:
    return request.app.state.pipeline
