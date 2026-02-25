#!/usr/bin/env python3
"""
Test script for RAG system components.
Verifies that everything works correctly before running the bot.

Usage:
    python scripts/test_rag_system.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import numpy as np
import gc
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def safe_remove(file_path):
    """Safely remove a file with retries for Windows."""
    if not os.path.exists(file_path):
        return
        
    for i in range(5):
        try:
            gc.collect()  # Force garbage collection to close file handles
            os.remove(file_path)
            return
        except PermissionError:
            if i < 4:
                time.sleep(0.5)
            else:
                print(f"⚠️ Could not remove {file_path} after retries")
                raise


def test_embeddings():
    """Test embedding service."""
    print_header("TEST 1: Embedding Service")
    
    try:
        from src.services.embeddings import EmbeddingService
        
        service = EmbeddingService()
        
        # Test single text
        test_text = "Абылай хан - қазақтың ұлы қолбасшысы"
        embedding = service.encode(test_text)[0]
        
        assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
        assert embedding.shape[0] == service.embedding_dim, "Wrong embedding dimension"
        
        print(f"✓ Single text encoding works")
        print(f"  - Embedding dimension: {embedding.shape[0]}")
        print(f"  - Embedding type: {type(embedding)}")
        
        # Test batch encoding
        texts = [
            "Қазақстан тарихы",
            "Python программалау",
            "Математика есептері"
        ]
        embeddings = service.encode(texts)
        
        assert embeddings.shape[0] == len(texts), "Wrong number of embeddings"
        print(f"✓ Batch encoding works ({len(texts)} texts)")
        
        # Test similarity
        emb1 = service.encode("Абылай хан")[0]
        emb2 = service.encode("Қазақ ханы")[0]
        emb3 = service.encode("Python код")[0]
        
        sim_related = service.compute_similarity(emb1, emb2)
        sim_unrelated = service.compute_similarity(emb1, emb3)
        
        print(f"✓ Similarity computation works")
        print(f"  - Related texts similarity: {sim_related:.3f}")
        print(f"  - Unrelated texts similarity: {sim_unrelated:.3f}")
        
        assert sim_related > sim_unrelated, "Related texts should be more similar"
        
        print("\n✅ Embedding Service: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Embedding Service: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunker():
    """Test chunking service."""
    print_header("TEST 2: Document Chunker")
    
    try:
        from src.services.chunker import DocumentChunker
        
        chunker = DocumentChunker(chunk_size=512, overlap=64)
        
        # Test with short text
        short_text = "Бұл қысқа мәтін."
        chunks = chunker.chunk_text(short_text)
        
        assert len(chunks) == 1, "Short text should produce 1 chunk"
        print(f"✓ Short text chunking works (1 chunk)")
        
        # Test with long text
        long_text = " ".join([
            "Қазақстан тарихы өте бай және қызықты.",
            "Көптеген ұлы тұлғалар қазақ жеріне келген.",
            "Олардың ішінде Абылай хан ерекше орын алады.",
            "Ол XVIII ғасырда өмір сүрген."
        ] * 50)  # Repeat to make it longer
        
        chunks = chunker.chunk_text(long_text)
        
        assert len(chunks) > 1, "Long text should produce multiple chunks"
        print(f"✓ Long text chunking works ({len(chunks)} chunks)")
        
        # Check overlap
        if len(chunks) > 1:
            # Some words from end of chunk 1 should appear in start of chunk 2
            chunk1_words = set(chunks[0]['content'].split()[-10:])
            chunk2_words = set(chunks[1]['content'].split()[:10])
            overlap_words = chunk1_words & chunk2_words
            
            assert len(overlap_words) > 0, "Chunks should have overlap"
            print(f"✓ Overlap detected: {len(overlap_words)} words")
        
        # Test document chunking
        document = {
            'id': 'test_doc',
            'content': long_text,
            'subject': 'тарих',
            'topic': 'test'
        }
        
        doc_chunks = chunker.chunk_document(document)
        assert len(doc_chunks) > 0, "Document chunking failed"
        assert 'metadata' in doc_chunks[0], "Chunk should have metadata"
        
        print(f"✓ Document chunking works")
        print(f"  - Total chunks: {len(doc_chunks)}")
        print(f"  - Metadata preserved: {list(doc_chunks[0]['metadata'].keys())}")
        
        print("\n✅ Document Chunker: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Document Chunker: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_db():
    """Test vector database."""
    print_header("TEST 3: Vector Database")
    
    try:
        from src.services.vector_db import VectorDB
        from src.services.embeddings import EmbeddingService
        
        # Use test database
        test_db_path = "test_documents.db"
        
        # Clean up if exists
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        db = VectorDB(db_path=test_db_path)
        embedding_service = EmbeddingService()
        
        # Add test document
        doc_id = "test_doc_1"
        db.add_document(
            doc_id=doc_id,
            source_file="test.json",
            metadata={'subject': 'тарих', 'topic': 'test'}
        )
        print(f"✓ Document added to database")
        
        # Add test chunks
        chunks = [
            {
                'content': 'Абылай хан - қазақтың ұлы қолбасшысы',
                'chunk_index': 0,
                'token_count': 10,
                'metadata': {'subject': 'тарих'}
            },
            {
                'content': 'Python - программалау тілі',
                'chunk_index': 1,
                'token_count': 8,
                'metadata': {'subject': 'информатика'}
            }
        ]
        
        # Generate embeddings
        embeddings = embedding_service.encode([c['content'] for c in chunks])
        
        # Add to database
        added = db.add_chunks(doc_id, chunks, embeddings)
        assert added == len(chunks), "Not all chunks were added"
        print(f"✓ Chunks added to database ({added} chunks)")
        
        # Test search
        query = "Қазақ ханы"
        query_embedding = embedding_service.encode(query)[0]
        
        results = db.search(query_embedding, top_k=2)
        
        assert len(results) > 0, "Search returned no results"
        assert 'similarity' in results[0], "Results missing similarity score"
        assert results[0]['similarity'] > 0, "Invalid similarity score"
        
        print(f"✓ Search works")
        print(f"  - Query: '{query}'")
        print(f"  - Results: {len(results)}")
        print(f"  - Top similarity: {results[0]['similarity']:.3f}")
        print(f"  - Top content: {results[0]['content'][:50]}...")
        
        # Test filters
        results_filtered = db.search(
            query_embedding,
            top_k=2,
            filters={'subject': 'тарих'}
        )
        
        print(f"✓ Filtered search works ({len(results_filtered)} results)")
        
        # Test stats
        stats = db.get_database_stats()
        assert stats['total_documents'] == 1, "Wrong document count"
        assert stats['total_chunks'] == 2, "Wrong chunk count"
        
        print(f"✓ Database stats work")
        print(f"  - Documents: {stats['total_documents']}")
        print(f"  - Chunks: {stats['total_chunks']}")
        
        # Clean up
        safe_remove(test_db_path)
        
        print("\n✅ Vector Database: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Vector Database: FAILED - {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on failure
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        return False


def test_full_pipeline():
    """Test complete indexing pipeline."""
    print_header("TEST 4: Full Indexing Pipeline")
    
    try:
        from src.services.indexer import DocumentIndexer
        import json
        import tempfile
        
        # Create temporary RAG directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test documents
            test_docs = [
                {
                    'id': 'history_test_1',
                    'content': 'Абылай хан XVIII ғасырда өмір сүрген ұлы қазақ қолбасшысы. '
                              'Ол үш жүзді біріктіруге үлкен үлес қосты.',
                    'subject': 'тарих',
                    'topic': 'Қазақстан тарихы',
                    'difficulty': 'medium'
                },
                {
                    'id': 'informatics_test_1',
                    'content': 'Python программалау тілі - бұл жоғары деңгейлі тіл. '
                              'Ол веб-әзірлеу, деректерді талдау және машиналық оқытуда кеңінен қолданылады.',
                    'subject': 'информатика',
                    'topic': 'программалау',
                    'difficulty': 'easy'
                }
            ]
            
            # Write test documents
            for doc in test_docs:
                file_path = os.path.join(temp_dir, f"{doc['id']}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Created {len(test_docs)} test documents")
            
            # Create test database
            test_db_path = "test_pipeline.db"
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
            
            # Initialize indexer
            indexer = DocumentIndexer(
                rag_directory=temp_dir,
                db_path=test_db_path,
                chunk_size=256,
                overlap=32
            )
            
            print(f"✓ Indexer initialized")
            
            # Run indexing
            stats = indexer.index_all(clear_existing=True)
            
            assert stats['success'], "Indexing failed"
            assert stats['documents_successful'] == len(test_docs), "Not all documents indexed"
            assert stats['total_chunks'] > 0, "No chunks created"
            
            print(f"✓ Indexing completed")
            print(f"  - Documents: {stats['documents_successful']}/{stats['documents_processed']}")
            print(f"  - Chunks: {stats['total_chunks']}")
            print(f"  - Time: {stats['elapsed_time']:.2f}s")
            
            # Test search on indexed data
            from src.services.embeddings import EmbeddingService
            from src.services.vector_db import VectorDB
            
            embedding_service = EmbeddingService()
            vector_db = VectorDB(db_path=test_db_path)
            
            # Search for historical content
            query = "Қазақ ханы"
            query_embedding = embedding_service.encode(query)[0]
            results = vector_db.search(query_embedding, top_k=3)
            
            assert len(results) > 0, "Search returned no results"
            print(f"✓ Search on indexed data works ({len(results)} results)")
            
            # Search with filter
            filtered_results = vector_db.search(
                query_embedding,
                top_k=3,
                filters={'subject': 'тарих'}
            )
            
            print(f"✓ Filtered search works ({len(filtered_results)} results)")
            
            # Clean up
            indexer.unload_model()
            os.remove(test_db_path)
        
        print("\n✅ Full Pipeline: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Full Pipeline: FAILED - {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on failure
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪 AsylBILIM RAG System Test Suite".center(60))
    print("Testing all components before production use\n")
    
    tests = [
        ("Embedding Service", test_embeddings),
        ("Document Chunker", test_chunker),
        ("Vector Database", test_vector_db),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️ Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ {test_name}: CRASHED - {e}")
            results[test_name] = False
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("\n" + "="*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! RAG system is ready.")
        print("\nNext steps:")
        print("  1. Run: python scripts/index_documents.py")
        print("  2. Start bot: python main.py")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please fix issues before using.")
        return 1


if __name__ == "__main__":
    sys.exit(main())