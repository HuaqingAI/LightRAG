from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.constants import (
    DUPLICATE_DISMISSED_METADATA_KEY,
    DUPLICATE_RETRYABLE_METADATA_KEY,
    FULL_DOCS_FORMAT_PENDING_PARSE,
)
from lightrag.kg.pipeline_ingress import PipelineIngressMessage
from lightrag.kg.shared_storage import get_pipeline_ingress
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _chunking(tokenizer, content, *args) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"retrydup-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


async def _publish_retry(rag: LightRAG) -> str:
    request_id = uuid4().hex
    ingress = await get_pipeline_ingress(rag.workspace)
    ingress.request_manual_retry(
        request_id,
        PipelineIngressMessage(kind="rescan", retry_failed=True, request_id=request_id),
    )
    return request_id


async def _make_retryable_duplicate(rag: LightRAG, staging_name: str) -> str:
    await rag.apipeline_enqueue_documents(input="old body", file_paths="same.txt")

    returned_track = await rag.apipeline_enqueue_documents(
        input="",
        file_paths="same.txt",
        source_files=staging_name,
        track_id="upload-new",
        docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
        parse_engine="legacy",
        retryable_duplicate_source="webui_upload",
    )
    assert returned_track == "upload-new"

    failed_docs = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
    duplicate_ids = [
        doc_id
        for doc_id, row in failed_docs.items()
        if (row.metadata or {}).get(DUPLICATE_RETRYABLE_METADATA_KEY) is True
    ]
    assert len(duplicate_ids) == 1
    return duplicate_ids[0]


def test_retryable_duplicate_keeps_full_docs_and_waits_while_original_exists(
    tmp_path, monkeypatch
):
    async def _run():
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        staging_name = "__lightrag_upload_test__same.txt"
        (input_dir / staging_name).write_text("new body", encoding="utf-8")

        rag = await _build_rag(tmp_path)
        try:
            duplicate_id = await _make_retryable_duplicate(rag, staging_name)

            full_doc = await rag.full_docs.get_by_id(duplicate_id)
            assert full_doc["parse_format"] == FULL_DOCS_FORMAT_PENDING_PARSE
            assert full_doc["file_path"] == "same.txt"
            assert full_doc["source_file"] == staging_name

            request_id = await _publish_retry(rag)
            assert await rag.apipeline_reset_failed_for_scan(request_id) is True

            row = await rag.doc_status.get_by_id(duplicate_id)
            assert row["status"] == DocStatus.FAILED
            assert row["metadata"][DUPLICATE_RETRYABLE_METADATA_KEY] is True
            assert row["metadata"]["original_doc_id"] == compute_mdhash_id(
                "same.txt", prefix="doc-"
            )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_retryable_duplicate_resets_after_original_source_is_absent(
    tmp_path, monkeypatch
):
    async def _run():
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        staging_name = "__lightrag_upload_test__same.txt"
        (input_dir / staging_name).write_text("new body", encoding="utf-8")

        rag = await _build_rag(tmp_path)
        try:
            duplicate_id = await _make_retryable_duplicate(rag, staging_name)
            original_id = compute_mdhash_id("same.txt", prefix="doc-")
            await rag.doc_status.delete([original_id])

            request_id = await _publish_retry(rag)
            assert await rag.apipeline_reset_failed_for_scan(request_id) is True

            row = await rag.doc_status.get_by_id(duplicate_id)
            assert row["status"] == DocStatus.PENDING
            metadata = row["metadata"]
            assert metadata["source_file"] == staging_name
            assert DUPLICATE_RETRYABLE_METADATA_KEY not in metadata
            assert "is_duplicate" not in metadata
            assert "duplicate_kind" not in metadata
            assert "original_doc_id" not in metadata
            assert "original_track_id" not in metadata
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_dismissed_retryable_duplicate_is_not_reset(tmp_path, monkeypatch):
    async def _run():
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        staging_name = "__lightrag_upload_test__same.txt"
        (input_dir / staging_name).write_text("new body", encoding="utf-8")

        rag = await _build_rag(tmp_path)
        try:
            duplicate_id = await _make_retryable_duplicate(rag, staging_name)
            original_id = compute_mdhash_id("same.txt", prefix="doc-")
            duplicate = await rag.doc_status.get_by_id(duplicate_id)
            await rag.doc_status.update_doc_status_fields(
                duplicate_id,
                {
                    "metadata": {
                        **duplicate["metadata"],
                        DUPLICATE_DISMISSED_METADATA_KEY: True,
                    }
                },
            )
            await rag.doc_status.delete([original_id])

            request_id = await _publish_retry(rag)
            assert await rag.apipeline_reset_failed_for_scan(request_id) is True

            row = await rag.doc_status.get_by_id(duplicate_id)
            assert row["status"] == DocStatus.FAILED
            assert row["metadata"][DUPLICATE_DISMISSED_METADATA_KEY] is True
            assert row["metadata"][DUPLICATE_RETRYABLE_METADATA_KEY] is True
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
