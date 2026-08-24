import json
import os
from datetime import datetime, timezone
from pathlib import Path


METRICS_FILE = Path(
    os.getenv("METRICS_FILE", "data/financial_metrics.json")
)


def _load() -> list[dict]:
    if not METRICS_FILE.exists():
        return []
    try:
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save(rows: list[dict]) -> None:
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_metrics(company: str, year: int | str | None, metrics: dict) -> None:
    """Persist the latest extracted metrics without requiring a database server."""
    rows = _load()
    year = str(year) if year is not None else ""

    rows = [
        row for row in rows
        if not (row.get("company") == company and str(row.get("year")) == year)
    ]

    rows.append(
        {
            "company": company,
            "year": year,
            **metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save(rows)

    print(f"Successfully saved metrics for {company} {year}.")


def get_metrics() -> list[dict]:
    """Return the latest saved metrics, ordered by company."""
    rows = _load()
    return sorted(rows, key=lambda row: row.get("company", ""))
