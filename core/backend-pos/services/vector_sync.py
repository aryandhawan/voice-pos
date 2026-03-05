"""
Vector Sync Service - JSON to Vector Pipeline
Handles real-time embedding and ChromaDB upserts from Module 1 webhook
"""

import os
import asyncio
import logging
from typing import List, Tuple, Optional
from datetime import datetime

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from schemas.revenue_sync import RevenueSyncPayload, MenuItemPayload, ComboRulePayload

logger = logging.getLogger(__name__)


class VectorSyncService:
    """
    Real-time vector synchronization service.
    Receives Module 1 JSON → Embeds → Upserts to ChromaDB
    """

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    BATCH_SIZE = 100

    def __init__(
        self,
        chroma_client: Optional[chromadb.Client] = None,
        embedder: Optional[SentenceTransformer] = None
    ):
        # Initialize ChromaDB (persistent storage)
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.chroma = chroma_client or chromadb.PersistentClient(path=db_path)

        # Load embedding model (lazy singleton pattern)
        if embedder is None:
            logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL}")
            self.embedder = SentenceTransformer(self.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        else:
            self.embedder = embedder

        # Get or create collections
        self.menu_collection = self.chroma.get_or_create_collection(
            name="menu_intelligence",
            metadata={"hnsw:space": "cosine", "description": "Menu items with BI metadata"}
        )

        self.combo_collection = self.chroma.get_or_create_collection(
            name="combo_rules",
            metadata={"hnsw:space": "cosine", "description": "Item pairing recommendations"}
        )

        self.upsell_collection = self.chroma.get_or_create_collection(
            name="upsell_candidates",
            metadata={"hnsw:space": "cosine", "description": "High-margin upsell items"}
        )

    async def process_revenue_sync(self, payload: RevenueSyncPayload) -> dict:
        """
        Main entry point: Process incoming revenue sync webhook
        Returns processing statistics
        """
        stats = {
            "sync_id": payload.sync_id,
            "menu_items_processed": 0,
            "upsell_items_indexed": 0,
            "combo_rules_indexed": 0,
            "deleted_count": 0,
            "errors": []
        }

        try:
            logger.info(f"Starting sync {payload.sync_id}: {len(payload.menu_items)} items")

            # Phase 1: Process menu items (all go to menu_intelligence)
            if payload.menu_items:
                await self._process_menu_items(payload.menu_items, payload.sync_id)
                stats["menu_items_processed"] = len(payload.menu_items)

                # Also index hidden stars to dedicated upsell collection
                hidden_stars = [item for item in payload.menu_items if item.hidden_star]
                if hidden_stars:
                    await self._process_upsell_candidates(hidden_stars, payload.sync_id)
                    stats["upsell_items_indexed"] = len(hidden_stars)

            # Phase 2: Process combo rules
            if payload.combo_rules:
                await self._process_combo_rules(payload.combo_rules)
                stats["combo_rules_indexed"] = len(payload.combo_rules)

            # Phase 3: Handle deletions
            if payload.deleted_items:
                self._delete_items(payload.deleted_items)
                stats["deleted_count"] = len(payload.deleted_items)

            logger.info(f"Sync {payload.sync_id} completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Sync {payload.sync_id} failed: {e}")
            stats["errors"].append(str(e))
            raise

    async def _process_menu_items(self, items: List[MenuItemPayload], sync_id: str):
        """Embed and upsert menu items to ChromaDB"""

        # Extract rich documents for embedding
        documents, metadatas, ids = self._build_documents(items, sync_id)

        # Generate embeddings (CPU-bound → run in threadpool)
        embeddings = await self._generate_embeddings_async(documents)

        # Batch upsert to ChromaDB
        await self._upsert_batch(
            collection=self.menu_collection,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"Upserted {len(ids)} items to menu_intelligence")

    async def _process_upsell_candidates(self, items: List[MenuItemPayload], sync_id: str):
        """Index hidden stars and high-margin items to dedicated upsell collection"""

        documents, metadatas, ids = self._build_documents(items, sync_id, is_upsell=True)
        embeddings = await self._generate_embeddings_async(documents)

        await self._upsert_batch(
            collection=self.upsell_collection,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"Upserted {len(ids)} upsell candidates")

    async def _process_combo_rules(self, rules: List[ComboRulePayload]):
        """Create joint embeddings for item pairs"""

        for rule in rules:
            # Retrieve embeddings for both items
            trigger_results = self.menu_collection.get(
                ids=[rule.trigger_item_id],
                include=["embeddings"]
            )

            rec_results = self.menu_collection.get(
                ids=[rule.recommendation_id],
                include=["embeddings"]
            )

            if trigger_results["embeddings"] and rec_results["embeddings"]:
                # Create averaged combo embedding
                trigger_emb = np.array(trigger_results["embeddings"][0])
                rec_emb = np.array(rec_results["embeddings"][0])
                combo_emb = ((trigger_emb + rec_emb) / 2).tolist()

                combo_id = f"combo_{rule.trigger_item_id}_{rule.recommendation_id}"

                self.combo_collection.upsert(
                    ids=[combo_id],
                    embeddings=[combo_emb],
                    documents=[f"Combo pair: {rule.trigger_item_id} + {rule.recommendation_id}"],
                    metadatas=[{
                        "trigger_id": rule.trigger_item_id,
                        "recommendation_id": rule.recommendation_id,
                        "lift_margin": rule.lift_margin,
                        "confidence": rule.confidence_score,
                        "type": "combo_rule"
                    }]
                )

    def _build_documents(
        self,
        items: List[MenuItemPayload],
        sync_id: str,
        is_upsell: bool = False
    ) -> Tuple[List[str], List[dict], List[str]]:
        """
        Build rich text documents from structured JSON for semantic embedding
        """
        documents = []
        metadatas = []
        ids = []

        for item in items:
            # Rich semantic text combining multiple fields
            doc_parts = [
                item.name,
                item.description if item.description else "",
                f"Category: {item.category}",
                f"Type: {item.category.replace('_', ' ')}",
            ]

            # Add semantic tags
            if item.upsell_tags:
                doc_parts.append(f"Tags: {', '.join(item.upsell_tags)}")

            if item.seasonal_tag:
                doc_parts.append(f"Season: {item.seasonal_tag}")

            # Add margin characteristics for semantic differentiation
            if item.contribution_margin > 150:
                doc_parts.append("Premium high-margin item")
            elif item.hidden_star:
                doc_parts.append("Under-promoted valuable item")

            if item.popularity_score > 0.7:
                doc_parts.append("Popular bestseller")
            elif item.popularity_score < 0.3:
                doc_parts.append("Niche specialty")

            document_text = ". ".join(filter(None, doc_parts))

            documents.append(document_text)
            ids.append(item.item_id)

            # Structured metadata for filtering
            metadata = {
                "item_id": item.item_id,
                "name": item.name,
                "category": item.category,
                "contribution_margin": float(item.contribution_margin),
                "margin_percent": float(item.margin_percent),
                "risk_flag": bool(item.risk_flag),
                "hidden_star": bool(item.hidden_star),
                "popularity_score": float(item.popularity_score),
                "seasonal_tag": item.seasonal_tag or "",
                "upsell_tags": ",".join(item.upsell_tags),
                "sync_id": sync_id,
                "is_upsell_collection": is_upsell,
                "last_updated": item.last_updated.isoformat() if item.last_updated else datetime.utcnow().isoformat()
            }

            metadatas.append(metadata)

        return documents, metadatas, ids

    async def _generate_embeddings_async(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings using threadpool for CPU-bound operation"""

        loop = asyncio.get_event_loop()

        def _encode():
            return self.embedder.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=32
            )

        embeddings = await loop.run_in_executor(None, _encode)
        return embeddings.tolist()

    async def _upsert_batch(
        self,
        collection: chromadb.Collection,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict]
    ):
        """Batch upsert with yielding for async cooperation"""

        for i in range(0, len(ids), self.BATCH_SIZE):
            batch_slice = slice(i, i + self.BATCH_SIZE)

            collection.upsert(
                ids=ids[batch_slice],
                embeddings=embeddings[batch_slice],
                documents=documents[batch_slice],
                metadatas=metadatas[batch_slice]
            )

            # Yield control to event loop
            await asyncio.sleep(0)

    def _delete_items(self, item_ids: List[str]):
        """Remove items from all collections"""
        for collection in [self.menu_collection, self.upsell_collection]:
            try:
                collection.delete(ids=item_ids)
            except Exception as e:
                logger.warning(f"Failed to delete from {collection.name}: {e}")

    # Query methods for Voice Intent

    async def semantic_search(
        self,
        query_text: str,
        n_results: int = 5,
        filter_hidden_stars: bool = False
    ) -> List[dict]:
        """
        Search menu items by semantic similarity
        Used for intent mapping from transcript
        """
        # Get query embedding
        query_emb = self.embedder.encode([query_text])[0].tolist()

        collection = self.upsell_collection if filter_hidden_stars else self.menu_collection

        where_filter = {"hidden_star": True} if filter_hidden_stars else None

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )

        items = []
        if results["ids"][0]:
            for i, item_id in enumerate(results["ids"][0]):
                items.append({
                    "item_id": item_id,
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "confidence": 1 - results["distances"][0][i]  # Convert distance to confidence
                })

        return items

    async def get_upsell_recommendations(
        self,
        item_ids: List[str],
        n_per_item: int = 2
    ) -> List[dict]:
        """Get semantic upsell recommendations for cart items"""

        recommendations = []

        for item_id in item_ids:
            # Get item embedding
            item_results = self.menu_collection.get(
                ids=[item_id],
                include=["embeddings"]
            )

            if not item_results["embeddings"]:
                continue

            # Search upsell collection for similar high-margin items
            upsell_results = self.upsell_collection.query(
                query_embeddings=item_results["embeddings"],
                n_results=n_per_item,
                where={"hidden_star": True},
                include=["metadatas", "distances"]
            )

            if upsell_results["ids"][0]:
                for i, rec_id in enumerate(upsell_results["ids"][0]):
                    if rec_id != item_id:  # Don't recommend the same item
                        recommendations.append({
                            "trigger_item": item_id,
                            "upsell_item_id": rec_id,
                            "upsell_metadata": upsell_results["metadatas"][0][i],
                            "confidence": 1 - upsell_results["distances"][0][i]
                        })

        return recommendations
