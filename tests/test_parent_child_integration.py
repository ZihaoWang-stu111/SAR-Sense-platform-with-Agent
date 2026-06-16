import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import get_vector_store_service
from rag.rag_service import RagSummarizeService


def main():
    vs = get_vector_store_service()
    fname = "SAR舰船检测概述.txt"
    if fname in vs.manifest:
        vs.delete_document(fname)

    new, upd, skip, rem = vs.load_document()
    print("load:", new, upd, skip, rem)

    entry = vs.manifest.get(fname, {})
    print("chunk_method:", entry.get("chunk_method"))
    print("parent_count:", entry.get("parent_count"))
    print("child_count:", entry.get("child_count"))
    print("docstore total:", vs.parent_docstore.count())

    rag = RagSummarizeService()
    query = "SAR舰船检测是什么"
    children = vs.get_retriever(query).invoke(query)
    print("children retrieved:", len(children))
    print("first child parent_id:", children[0].metadata.get("parent_id") if children else None)

    parents = rag.parent_resolver.resolve(children)
    print("parents resolved:", len(parents))
    if parents:
        print("parent len:", len(parents[0].page_content))
        print("parent preview:", parents[0].page_content[:120].replace("\n", " "))


if __name__ == "__main__":
    main()
