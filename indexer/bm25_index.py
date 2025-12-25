from whoosh.index import create_in, open_dir,exists_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import os

class BM25Indexer:
    """
    BM25 keyword based index for multimodal context (text + image caption)
    """

    def __init__(self, index_dir="index/bm25"):
        self.index_dir = index_dir
        self.schema = Schema(
            file = ID(stored=True, unique=True),
            content = TEXT(stored = True)
        )
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        
        # Check if index exists
        if exists_in(index_dir):
            self.ix = open_dir(index_dir)
        else:
            self.ix = create_in(index_dir, self.schema)
    
    def add_documents(self, docs):
        """
        docs : list of {"file":...,  "content":...}
        """

        writer = self.ix.writer()
        for doc in docs:
            writer.update_document(file=doc["file"], content=doc["content"])
        writer.commit()

    
    def search(self, query, top_k = 10):
        qp = QueryParser("content", schema=self.ix.schema)
        q = qp.parse(query)
        with self.ix.searcher() as searcher:
            results = searcher.search(q, limit=top_k)
            return [{"file": r["file"], "content": r["content"]} for r in results]
        
