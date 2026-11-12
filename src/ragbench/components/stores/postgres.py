"""Chunk index in PostgreSQL: pgvector for dense search, full-text for lexical, RRF for hybrid.

Every index is one table. Search is a single SQL statement in all three modes; hybrid search
fuses the dense and the lexical ranking with reciprocal rank fusion inside the database.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from psycopg import sql

from ragbench.components.postgres import Database, shared_database
from ragbench.core.errors import StoreError
from ragbench.core.hashing import stable_hash
from ragbench.core.registries import register_store
from ragbench.core.resources import Resources
from ragbench.core.types import Chunk, Question, RetrievedChunk, SearchMode, Span

#: pgvector indexes `vector` up to 2,000 dimensions and `halfvec` up to 4,000.
_MAX_VECTOR_INDEX_DIMENSIONS = 2000
_MAX_HALFVEC_INDEX_DIMENSIONS = 4000
_INSERT_BATCH = 500

logger = logging.getLogger(__name__)

_DENSE = """
dense AS (
    SELECT id, distance, row_number() OVER (ORDER BY distance, id) AS position
    FROM (
        SELECT id, {embedding} <=> {query_vector} AS distance
        FROM {table}
        ORDER BY distance
        LIMIT %(depth)s
    ) AS nearest
)"""

_LEXICAL = """
terms AS (
    SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS query
    FROM unnest(tsvector_to_array(to_tsvector('english', %(query)s))) AS lexeme
),
lexical AS (
    SELECT id, relevance, row_number() OVER (ORDER BY relevance DESC, id) AS position
    FROM (
        SELECT c.id, ts_rank(c.tsv, terms.query, 1) AS relevance
        FROM {table} AS c, terms
        WHERE c.tsv @@ terms.query
        ORDER BY relevance DESC, c.id
        LIMIT %(depth)s
    ) AS matching
)"""

_FUSED = {
    "dense": "SELECT id, (1 - distance)::float8 AS score FROM dense",
    "lexical": "SELECT id, relevance::float8 AS score FROM lexical",
    # Reciprocal rank fusion: a chunk scores 1 / (k + rank) in each list that contains it.
    "hybrid": (
        "SELECT COALESCE(dense.id, lexical.id) AS id, "
        "(COALESCE(1.0 / (%(rrf_k)s + dense.position), 0) "
        "+ COALESCE(1.0 / (%(rrf_k)s + lexical.position), 0))::float8 AS score "
        "FROM dense FULL OUTER JOIN lexical ON dense.id = lexical.id"
    ),
}

_SEARCH = """
WITH {sources},
fused AS ({fused}),
passages AS (
    SELECT c.document_id, c.ret_start, c.ret_end, c.ret_text, fused.score, c.id,
           row_number() OVER (
               PARTITION BY c.document_id, c.ret_start, c.ret_end
               ORDER BY fused.score DESC, c.id
           ) AS copy
    FROM fused JOIN {table} AS c ON c.id = fused.id
)
SELECT document_id, ret_start, ret_end, ret_text, score
FROM passages
WHERE copy = 1
ORDER BY score DESC, id
LIMIT %(top_k)s"""


@register_store("postgres")
class PostgresStore:
    """Stores chunks in PostgreSQL and searches them in dense, lexical or hybrid mode.

    Args:
        resources: Shared dependencies; provides the connection pool.
        url: PostgreSQL connection URL.
        schema: Schema that holds the index tables.
        rrf_k: Constant ``k`` of reciprocal rank fusion.
        depth_factor: Each ranking is read ``depth_factor * top_k`` deep (at least 50) before
            fusion and before chunks that share a parent are collapsed.
    """

    def __init__(
        self,
        resources: Resources,
        url: str,
        schema: str = "ragbench",
        rrf_k: int = 60,
        depth_factor: int = 5,
    ) -> None:
        self._database: Database = shared_database(resources, url, schema)
        self._catalog = self._database.table("indexes")
        self._rrf_k = rrf_k
        self._depth_factor = depth_factor
        self._catalog_ready = False
        self._dimensions: dict[str, int | None] = {}

    def _table(self, index: str) -> sql.Identifier:
        return self._database.table(f"idx_{stable_hash(index, length=20)}")

    async def _ensure_catalog(self) -> None:
        if self._catalog_ready:
            return
        async with self._database.connection() as connection:
            await connection.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} ("
                    "name text PRIMARY KEY, dimensions integer, "
                    "complete boolean NOT NULL DEFAULT false, "
                    "created_at timestamptz NOT NULL DEFAULT now())"
                ).format(self._catalog)
            )
        self._catalog_ready = True

    async def _index_dimensions(self, index: str) -> int | None:
        if index not in self._dimensions:
            await self._ensure_catalog()
            async with self._database.connection() as connection:
                cursor = await connection.execute(
                    sql.SQL("SELECT dimensions FROM {} WHERE name = %s").format(self._catalog),
                    (index,),
                )
                row = await cursor.fetchone()
            if row is None:
                raise StoreError(f"index {index!r} does not exist")
            self._dimensions[index] = int(row[0]) if isinstance(row[0], int) else None
        return self._dimensions[index]

    async def has_index(self, index: str) -> bool:
        """Return whether ``index`` exists and was completely written."""
        await self._ensure_catalog()
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                sql.SQL("SELECT complete FROM {} WHERE name = %s").format(self._catalog),
                (index,),
            )
            row = await cursor.fetchone()
        return row is not None and bool(row[0])

    async def create_index(self, index: str, *, dimensions: int | None) -> None:
        """Create ``index`` empty, dropping any earlier version of it."""
        await self.drop_index(index)
        table = self._table(index)
        embedding = (
            sql.SQL("vector({})").format(sql.Literal(dimensions))
            if dimensions is not None
            else sql.SQL("vector")
        )
        async with self._database.connection() as connection, connection.transaction():
            await connection.execute(
                sql.SQL(
                    "CREATE TABLE {} ("
                    "id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                    "document_id text NOT NULL, "
                    "span_start integer NOT NULL, span_end integer NOT NULL, "
                    "text text NOT NULL, "
                    "ret_start integer NOT NULL, ret_end integer NOT NULL, "
                    "ret_text text NOT NULL, "
                    "tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED, "
                    "embedding {})"
                ).format(table, embedding)
            )
            await connection.execute(
                sql.SQL("INSERT INTO {} (name, dimensions) VALUES (%s, %s)").format(self._catalog),
                (index, dimensions),
            )
        self._dimensions[index] = dimensions

    async def add_chunks(
        self, index: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]] | None
    ) -> None:
        """Append ``chunks`` and their ``vectors`` to ``index``.

        Raises:
            StoreError: If the index does not exist, or vectors are missing, superfluous or of
                the wrong length.
        """
        dimensions = await self._index_dimensions(index)
        if (dimensions is None) != (vectors is None):
            raise StoreError(f"index {index!r}: vectors must be given exactly for dense indexes")
        if vectors is not None:
            if len(vectors) != len(chunks):
                raise StoreError(f"index {index!r}: {len(chunks)} chunks, {len(vectors)} vectors")
            if any(len(vector) != dimensions for vector in vectors):
                raise StoreError(f"index {index!r}: vectors must have {dimensions} dimensions")
        statement = sql.SQL(
            "INSERT INTO {} (document_id, span_start, span_end, text, ret_start, ret_end, "
            "ret_text, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)"
        ).format(self._table(index))
        rows = [
            (
                chunk.document_id,
                chunk.span.start,
                chunk.span.end,
                chunk.text,
                chunk.returned_span.start,
                chunk.returned_span.end,
                chunk.returned_text,
                _vector_literal(vectors[position]) if vectors is not None else None,
            )
            for position, chunk in enumerate(chunks)
        ]
        async with self._database.connection() as connection, connection.cursor() as cursor:
            for start in range(0, len(rows), _INSERT_BATCH):
                await cursor.executemany(statement, rows[start : start + _INSERT_BATCH])

    def _embedding_expression(self, dimensions: int) -> sql.Composable:
        if dimensions > _MAX_VECTOR_INDEX_DIMENSIONS:
            return sql.SQL("(embedding::halfvec({}))").format(sql.Literal(dimensions))
        return sql.SQL("embedding")

    async def finish_index(self, index: str) -> None:
        """Build the full-text and HNSW indexes of ``index`` and mark it complete."""
        dimensions = await self._index_dimensions(index)
        table = self._table(index)
        async with self._database.connection() as connection:
            await connection.execute(sql.SQL("CREATE INDEX ON {} USING gin (tsv)").format(table))
            await connection.execute(
                sql.SQL("CREATE INDEX ON {} (document_id, ret_start, ret_end)").format(table)
            )
            if dimensions is not None and dimensions <= _MAX_HALFVEC_INDEX_DIMENSIONS:
                operator_class = (
                    "halfvec_cosine_ops"
                    if dimensions > _MAX_VECTOR_INDEX_DIMENSIONS
                    else "vector_cosine_ops"
                )
                await connection.execute(
                    sql.SQL("CREATE INDEX ON {} USING hnsw ({} {})").format(
                        table, self._embedding_expression(dimensions), sql.SQL(operator_class)
                    )
                )
            elif dimensions is not None:
                logger.warning(
                    "index %s has %d dimensions, more than pgvector can index; "
                    "dense search will scan every row",
                    index,
                    dimensions,
                )
            await connection.execute(sql.SQL("ANALYZE {}").format(table))
            await connection.execute(
                sql.SQL("UPDATE {} SET complete = true WHERE name = %s").format(self._catalog),
                (index,),
            )

    async def search(
        self,
        index: str,
        *,
        mode: SearchMode,
        query: str,
        vector: Sequence[float] | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` best passages for the query, best first.

        Raises:
            StoreError: If the index does not exist, or a dense or hybrid search is asked of a
                lexical-only index or without a query vector.
        """
        dimensions = await self._index_dimensions(index)
        table = self._table(index)
        sources = []
        parameters: dict[str, object] = {
            "query": query,
            "top_k": top_k,
            "depth": max(self._depth_factor * top_k, 50),
            "rrf_k": self._rrf_k,
        }
        if mode in ("dense", "hybrid"):
            if dimensions is None or vector is None:
                raise StoreError(f"{mode} search on {index!r} needs a dense index and a vector")
            if len(vector) != dimensions:
                raise StoreError(f"index {index!r}: the query vector must have {dimensions} values")
            cast = "halfvec" if dimensions > _MAX_VECTOR_INDEX_DIMENSIONS else "vector"
            parameters["vector"] = _vector_literal(vector)
            sources.append(
                sql.SQL(_DENSE).format(
                    table=table,
                    embedding=self._embedding_expression(dimensions),
                    query_vector=sql.SQL("%(vector)s::{}").format(sql.SQL(cast)),
                )
            )
        if mode in ("lexical", "hybrid"):
            sources.append(sql.SQL(_LEXICAL).format(table=table))
        statement = sql.SQL(_SEARCH).format(
            sources=sql.SQL(",").join(sources), fused=sql.SQL(_FUSED[mode]), table=table
        )
        async with self._database.connection() as connection, connection.transaction():
            # Let HNSW look at as many candidates as the query asks for (the default is 40).
            await connection.execute(
                sql.SQL("SET LOCAL hnsw.ef_search = {}").format(
                    sql.Literal(min(1000, int(str(parameters["depth"]))))
                )
            )
            cursor = await connection.execute(statement, parameters)
            rows = await cursor.fetchall()
        results = []
        for rank, (document_id, start, end, text, score) in enumerate(rows, start=1):
            results.append(
                RetrievedChunk(
                    document_id=str(document_id),
                    span=Span(int(str(start)), int(str(end))),
                    text=str(text),
                    score=float(str(score)),
                    rank=rank,
                )
            )
        return results

    async def relevant_keys(self, index: str, question: Question) -> frozenset[str]:
        """Return the keys of every passage of ``index`` that overlaps a gold span."""
        await self._index_dimensions(index)
        overlaps = sql.SQL(" OR ").join(
            sql.SQL("(ret_start < {} AND ret_end > {})").format(
                sql.Literal(span.end), sql.Literal(span.start)
            )
            for span in question.spans
        )
        statement = sql.SQL(
            "SELECT DISTINCT document_id, ret_start, ret_end FROM {} "
            "WHERE document_id = %s AND ({})"
        ).format(self._table(index), overlaps)
        if not question.spans:
            return frozenset()
        async with self._database.connection() as connection:
            cursor = await connection.execute(statement, (question.document_id,))
            rows = await cursor.fetchall()
        return frozenset(f"{document_id}:{start}-{end}" for document_id, start, end in rows)

    async def count(self, index: str) -> int:
        """Return the number of chunks stored in ``index``."""
        await self._index_dimensions(index)
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                sql.SQL("SELECT count(*) FROM {}").format(self._table(index))
            )
            row = await cursor.fetchone()
        return int(str(row[0])) if row else 0

    async def drop_index(self, index: str) -> None:
        """Delete ``index`` if it exists."""
        await self._ensure_catalog()
        async with self._database.connection() as connection, connection.transaction():
            await connection.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self._table(index)))
            await connection.execute(
                sql.SQL("DELETE FROM {} WHERE name = %s").format(self._catalog), (index,)
            )
        self._dimensions.pop(index, None)


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"
