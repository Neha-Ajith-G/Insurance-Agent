"""
Local ChromaDB Vector Store with Ollama Local Embedding Function & Fallback Support.
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    logger.warning("chromadb package not found; vectorstore will use local in-memory fallback index.")


class OllamaEmbeddingFunction:
    def __init__(
        self,
        base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model: str = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _get_single_embedding(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if embedding and isinstance(embedding, list):
                return embedding
        except Exception as e:
            logger.debug("Ollama embedding call for model '%s' failed: %s", self.model, e)

        return self._fallback_hash_embedding(text)

    def _fallback_hash_embedding(self, text: str, dim: int = 64) -> list[float]:
        vec = [0.0] * dim
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec
        for word in words:
            h = hash(word) % dim
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, input: list[str]) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        return [self._get_single_embedding(t) for t in input]


class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "insurance_chunks",
        base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        embed_model: str = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embed_fn = OllamaEmbeddingFunction(base_url=base_url, model=embed_model)
        self.client = None
        self.collection = None
        self._fallback_records: list[dict[str, Any]] = []

        if HAS_CHROMADB:
            try:
                os.makedirs(persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_directory)
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embed_fn, # type: ignore
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("ChromaDB vector store initialized at '%s'", persist_directory)
            except Exception as e:
                logger.warning("Failed to initialize ChromaDB PersistentClient: %s", e)
                self.client = None
                self.collection = None

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[Chunk],
        filename: str,
        user_id: str = "",
    ) -> int:
        if not chunks:
            return 0

        ids = [f"{doc_id}_{c.chunk_id}" for c in chunks]
        documents = [c.vector_text for c in chunks]
        metadatas = []
        for c in chunks:
            m = c.metadata.model_dump()
            m["doc_id"] = doc_id
            m["filename"] = filename
            m["user_id"] = user_id
            m["breadcrumbs"] = " > ".join(c.metadata.section_breadcrumbs)
            m["vector_token_count"] = c.vector_token_count
            cleaned_m = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned_m[k] = v
                elif v is None:
                    cleaned_m[k] = ""
                elif isinstance(v, list):
                    cleaned_m[k] = " > ".join(str(x) for x in v)
                else:
                    cleaned_m[k] = str(v)
            metadatas.append(cleaned_m)

        if self.collection is not None:
            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                logger.info("Indexed %d chunks into ChromaDB for file '%s'", len(chunks), filename)
                return len(chunks)
            except Exception as e:
                logger.warning("ChromaDB upsert failed, storing in fallback index: %s", e)

        for cid, doc_text, meta, chk in zip(ids, documents, metadatas, chunks):
            emb = self.embed_fn._get_single_embedding(doc_text)
            self._fallback_records.append({
                "id": cid,
                "document": doc_text,
                "metadata": meta,
                "embedding": emb,
                "raw_content": chk.content,
            })
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        where_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        if self.collection is not None:
            try:
                kwargs: dict[str, Any] = {"query_texts": [query], "n_results": top_k}
                if where_filter:
                    kwargs["where"] = where_filter
                results = self.collection.query(**kwargs)

                matches = []
                if results and "documents" in results and results["documents"]:
                    docs = results["documents"][0]
                    ids = results["ids"][0] if "ids" in results else []
                    metas = results["metadatas"][0] if "metadatas" in results else []
                    dists = results["distances"][0] if "distances" in results else []

                    for doc, cid, meta, dist in zip(docs, ids, metas, dists):
                        score = round(max(0.0, 1.0 - float(dist)), 4) if dist is not None else 1.0
                        matches.append({
                            "id": cid,
                            "document": doc,
                            "metadata": meta,
                            "score": score,
                        })
                return matches
            except Exception as e:
                logger.warning("ChromaDB query failed: %s", e)

        query_vec = self.embed_fn._get_single_embedding(query)
        scored = []
        for rec in self._fallback_records:
            if where_filter:
                match_filter = all(rec["metadata"].get(k) == v for k, v in where_filter.items())
                if not match_filter:
                    continue
            vec = rec["embedding"]
            dot = sum(q * v for q, v in zip(query_vec, vec))
            q_norm = math.sqrt(sum(q * q for q in query_vec)) or 1.0
            v_norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            sim = dot / (q_norm * v_norm)
            scored.append((sim, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = []
        for sim, rec in scored[:top_k]:
            top_matches.append({
                "id": rec["id"],
                "document": rec["document"],
                "metadata": rec["metadata"],
                "score": round(sim, 4),
            })
        return top_matches

    def get_stats(self) -> dict[str, Any]:
        count = 0
        if self.collection is not None:
            try:
                count = self.collection.count()
            except Exception:
                count = 0
        else:
            count = len(self._fallback_records)

        return {
            "has_chromadb": HAS_CHROMADB,
            "vector_count": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }
