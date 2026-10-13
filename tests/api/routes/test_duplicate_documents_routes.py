from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_base = importlib.import_module("lightrag.base")
_constants = importlib.import_module("lightrag.constants")
sys.argv = _original_argv

DocStatus = _base.DocStatus
DUPLICATE_DISMISSED_METADATA_KEY = _constants.DUPLICATE_DISMISSED_METADATA_KEY
DUPLICATE_RETRYABLE_METADATA_KEY = _constants.DUPLICATE_RETRYABLE_METADATA_KEY
DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes
DismissDuplicateDocumentsRequest = _document_routes.DismissDuplicateDocumentsRequest

pytestmark = pytest.mark.offline


class _DuplicateDocStatus:
    supports_strict_point_reads = True

    def __init__(self, rows: dict[str, dict]):
        self.rows = rows
        self.updated: list[tuple[str, dict]] = []

    async def get_docs_by_statuses(self, statuses, strict: bool = False):
        wanted = {
            status.value if isinstance(status, DocStatus) else str(status)
            for status in statuses
        }
        return {
            doc_id: row
            for doc_id, row in self.rows.items()
            if (
                row.get("status").value
                if isinstance(row.get("status"), DocStatus)
                else str(row.get("status"))
            )
            in wanted
        }

    async def get_by_id_strict(self, doc_id: str):
        return self.rows.get(doc_id)

    async def update_doc_status_fields(
        self, doc_id: str, fields: dict, *, missing_ok: bool = False
    ):
        row = self.rows.get(doc_id)
        if row is None:
            if missing_ok:
                return
            raise KeyError(doc_id)
        row.update(fields)
        self.updated.append((doc_id, fields))


class _DuplicateRag:
    def __init__(self, rows: dict[str, dict]):
        self.workspace = f"dup-route-{uuid4().hex}"
        self.doc_status = _DuplicateDocStatus(rows)


def _row(
    *,
    status: DocStatus = DocStatus.FAILED,
    file_path: str = "same.txt",
    metadata: dict | None = None,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "status": status,
        "content_summary": "summary",
        "content_length": 10,
        "chunks_count": 0,
        "created_at": now,
        "updated_at": now,
        "file_path": file_path,
        "track_id": "track",
        "metadata": metadata or {},
    }


def _retryable_duplicate(original_doc_id: str, **metadata: object) -> dict:
    return _row(
        metadata={
            "is_duplicate": True,
            "duplicate_kind": "filename",
            DUPLICATE_RETRYABLE_METADATA_KEY: True,
            "original_doc_id": original_doc_id,
            **metadata,
        }
    )


def _route_endpoint(router, name: str):
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == name
    ][-1]


@pytest.fixture(autouse=True)
def _shared_storage():
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    yield


async def test_list_duplicate_documents_groups_retryable_rows_with_existing_original(
    tmp_path,
):
    rows = {
        "doc-old": _row(status=DocStatus.PROCESSED, file_path="same.txt"),
        "dup-1": _retryable_duplicate("doc-old"),
        "dup-dismissed": _retryable_duplicate(
            "doc-old", **{DUPLICATE_DISMISSED_METADATA_KEY: True}
        ),
        "dup-orphan": _retryable_duplicate("doc-missing"),
        "dup-content": _row(
            metadata={
                "is_duplicate": True,
                "duplicate_kind": "content_hash",
                DUPLICATE_RETRYABLE_METADATA_KEY: True,
                "original_doc_id": "doc-old",
            }
        ),
    }
    rag = _DuplicateRag(rows)
    router = create_document_routes(rag, DocumentManager(str(tmp_path)))

    response = await _route_endpoint(router, "list_duplicate_documents")()

    assert response.total_groups == 1
    assert response.total_duplicates == 1
    assert response.groups[0].original.id == "doc-old"
    assert response.groups[0].duplicate_doc_ids == ["dup-1"]
    assert [duplicate.id for duplicate in response.groups[0].duplicates] == ["dup-1"]


async def test_dismiss_duplicate_documents_marks_matching_retryable_rows(tmp_path):
    rows = {
        "doc-old": _row(status=DocStatus.PROCESSED, file_path="same.txt"),
        "dup-1": _retryable_duplicate("doc-old"),
        "dup-2": _retryable_duplicate("doc-old"),
        "dup-cleared": _retryable_duplicate(
            "doc-cleared", **{DUPLICATE_DISMISSED_METADATA_KEY: True}
        ),
        "dup-other": _retryable_duplicate("doc-other"),
    }
    rag = _DuplicateRag(rows)
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    router = create_document_routes(rag, DocumentManager(str(tmp_path)))

    response = await _route_endpoint(router, "dismiss_duplicate_documents")(
        DismissDuplicateDocumentsRequest(
            original_doc_ids=["doc-old", "doc-cleared", "doc-missing"]
        )
    )

    assert response.cleared == 2
    assert response.already_cleared == 1
    assert response.not_found == 1
    assert rows["dup-1"]["metadata"][DUPLICATE_DISMISSED_METADATA_KEY] is True
    assert rows["dup-2"]["metadata"][DUPLICATE_DISMISSED_METADATA_KEY] is True
    assert rows["dup-other"]["metadata"].get(DUPLICATE_DISMISSED_METADATA_KEY) is None
    assert {doc_id for doc_id, _ in rag.doc_status.updated} == {"dup-1", "dup-2"}
