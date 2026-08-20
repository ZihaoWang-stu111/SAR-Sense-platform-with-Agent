import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import get_vector_store_service
from rag.rag_service import RagSummarizeService
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path


def main():
    vs = get_vector_store_service()
    fname = "SAR舰船检测概述.txt"
    if vs.knowledge_repository.get_by_filename(fname) is not None:
        vs.delete_document(fname)

    path = os.path.join(get_abs_path(chroma_conf["data_path"]), fname)
    new, upd, skip, rem = vs.load_documents([path])
    print("load:", new, upd, skip, rem)

    document = vs.knowledge_repository.get_by_filename(fname)
    print("chunk_method:", document.chunk_method if document else None)
    print("parent_count:", document.parent_count if document else None)
    print("child_count:", document.child_count if document else None)
    print("docstore total:", vs.parent_docstore.count())

    rag = RagSummarizeService()
    query = "SAR舰船检测是什么"
    children = vs.retrieve(query)
    print("children retrieved:", len(children))
    print("first child parent_id:", children[0].metadata.get("parent_id") if children else None)

    parents = rag.parent_resolver.resolve(children)
    print("parents resolved:", len(parents))
    if parents:
        print("parent len:", len(parents[0].page_content))
        print("parent preview:", parents[0].page_content[:120].replace("\n", " "))


if __name__ == "__main__":
    main()
