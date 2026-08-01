# RAG File RBAC

This project uses file-level role based access control for the shared knowledge base.

## Roles

- `admin`: manages users, uploads/deletes files, edits file visibility, and can retrieve all documents.
- `researcher`, `business`, `guest`: can only list, download, and retrieve files whose ACL contains their role.

## Visibility Rule

Knowledge file ACL is stored in MySQL table `knowledge_documents`.

- `allowed_roles = []`: admin-only.
- `allowed_roles = ["researcher"]`: visible to admin and researcher.
- Admin is always allowed, even if the role list is empty.

MySQL is the only runtime source of truth. Existing local JSON data has completed its one-time cutover; new installations create ACL records through the current upload flow and do not depend on legacy JSON files.

## Request Flow

1. `api/routers/chat.py` reads the current user role from DB.
2. It resolves `allowed_doc_ids` through `crud.knowledge_acl.get_allowed_doc_ids`.
3. `ReactAgent.execute_stream(..., user_context=...)` passes those ids into LangGraph runtime context.
4. `rag_summarize(query, runtime)` reads `runtime.context["allowed_doc_ids"]`.
5. `RagSummarizeService` passes the ids into retrieval, rerank, and parent resolution.

The model never receives role or doc_id as a tool argument, so it cannot override permissions.

## Retrieval Chain

The hybrid retrieval chain still uses LangChain `EnsembleRetriever`.

- Vector side: Chroma retriever uses metadata filter `doc_id in allowed_doc_ids`.
- BM25 side: `FilteredBM25Retriever` wraps LangChain `BM25Retriever`, scores the full BM25 index, but only ranks documents whose `doc_id` is allowed.
- Rerank side: rerank only receives already-authorized child chunks, then filters again defensively.
- Parent-child side: `ParentChildResolver` checks `doc_id` again after parent lookup.
- Generation side: vector and BM25 retrieval are restricted to active chunk ids from MySQL, so failed or superseded uploads cannot leak into results.

This avoids the unsafe pattern of “full-library top-k, then filter”, which can hide authorized weak matches behind unauthorized strong matches.

## Storage And Updates

- `knowledge_documents` stores the active document generation, storage key, chunk ids, ACL, and ingestion status.
- `parent_chunks` stores parent text and metadata; Chroma stores child vectors.
- Uploaded source files use immutable version paths under `data/.knowledge_versions/<uuid>/`.
- A successful update activates the complete new generation before old vectors, parent chunks, and source files are cleaned up.
- Knowledge writes share the Redis knowledge lock so upload, delete, and permission changes do not race.
- MySQL, Chroma, and the filesystem do not share one transaction. A forced stop can leave inactive vectors, parent rows, or version files; active-id filtering keeps them out of retrieval, while maintenance cleanup remains an operational task.

## Validation

```bash
python -m compileall api agent rag crud schemas utils tests
python -m unittest tests.test_rag_acl_retrieval -v
```

Manual checks:

- Admin uploads a file as admin-only; guest cannot list, download, or retrieve it.
- Admin grants `researcher`; researcher can list, download, and retrieve it.
- Admin revokes the role; the next request stops retrieving it.
