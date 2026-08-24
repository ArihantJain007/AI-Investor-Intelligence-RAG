from fastapi import APIRouter

from database.json_store import get_metrics

router = APIRouter()


@router.get("/metrics")
def metrics():
    return get_metrics()
