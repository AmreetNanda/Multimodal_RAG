import argparse
import pickle
from crawler.pdf_crawler import PDFCrawler
from crawler.image_crawler import ImageCrawler
from extractor.pdf_text_extractor import PDFTextExtractor
from extractor.image_object_extractor import ImageObjectExtractor
from indexer.bm25_index import BM25Indexer
from indexer.vector_index import VectorIndexer
from indexer.hybrid_index import HybridIndexer
from utils.logger import get_logger
from tqdm import tqdm

logger = get_logger("IngestScript")

def main(args):
    
    #1. Crawl PDFs
    pdf_crawler = PDFCrawler(args.pdf_dirs)
    pdf_files = pdf_crawler.crawl()

    pdf_extractor = PDFTextExtractor()
    pdf_docs = [pdf_extractor.extract_text(f) for f in tqdm(pdf_files)]


    #2. Crawl Images
    img_crawler = ImageCrawler(args.image_dirs)
    image_files = img_crawler.crawl()

    img_extractor = ImageObjectExtractor()
    img_docs = [img_extractor.extract(f) for f in tqdm(image_files)]


    #3. Combine Documents
    all_docs = pdf_docs + img_docs

    #4. Build BM25 and Vector indices 
    bm25_indexer = BM25Indexer(index_dir = args.bm25_index)
    bm25_indexer.add_documents(all_docs)

    vector_indexer = VectorIndexer()
    vector_indexer.add_documents(all_docs)


    # Save the Hybrid index
    hybrid_indexer = HybridIndexer(bm25_indexer, vector_indexer)
    with open(args.output_file, "wb") as f:
        pickle.dump(hybrid_indexer, f)
    logger.info(f"Hybrod index saved to {args.output_file}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dirs", nargs="+", required=True)
    parser.add_argument("--image_dirs", nargs="+", required=True)
    parser.add_argument("--bm25_index", type=str, default="index/bm25")
    parser.add_argument("--output_file", type=str, default="index/hybrid/hybrid_index.pkl")
    args = parser.parse.args()
    main(args)



