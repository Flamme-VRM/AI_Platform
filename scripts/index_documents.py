#!/usr/bin/env python3
"""
Offline indexing script for AsylBILIM RAG system.
Run this script to index all documents before starting the bot.

Usage:
    python scripts/index_documents.py                    # Index new documents
    python scripts/index_documents.py --clear            # Clear and re-index all
    python scripts/index_documents.py --status           # Show indexing status
    python scripts/index_documents.py --reindex DOC_ID   # Re-index specific document
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.indexer import DocumentIndexer
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     AsylBILIM Document Indexer                          ║
║     Offline Semantic Search Indexing                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_stats(stats: dict):
    """Pretty print indexing statistics."""
    print("\n" + "="*60)
    print("📊 INDEXING STATISTICS")
    print("="*60)
    
    if stats.get('success'):
        print(f"✓ Status:              Success")
        print(f"✓ Documents processed: {stats['documents_processed']}")
        print(f"✓ Documents indexed:   {stats['documents_successful']}")
        if stats['documents_failed'] > 0:
            print(f"✗ Documents failed:    {stats['documents_failed']}")
        print(f"✓ Total chunks:        {stats['total_chunks']}")
        print(f"✓ Total tokens:        {stats['total_tokens']:,}")
        print(f"✓ Avg chunk size:      {stats['avg_chunk_size']:.1f} tokens")
        print(f"✓ Time elapsed:        {stats['elapsed_time']:.2f} seconds")
        
        if stats.get('subjects'):
            print(f"\n📚 Subjects indexed:")
            for subject, count in stats['subjects'].items():
                print(f"   • {subject}: {count} documents")
    else:
        print(f"✗ Status: Failed")
        print(f"✗ Error: {stats.get('error', 'Unknown error')}")
    
    print("="*60 + "\n")


def print_status(status: dict):
    """Pretty print indexing status."""
    print("\n" + "="*60)
    print("📊 CURRENT INDEXING STATUS")
    print("="*60)
    
    print(f"RAG files available:   {status['rag_files_available']}")
    print(f"Documents indexed:     {status['documents_indexed']}")
    print(f"Needs indexing:        {'Yes ⚠️' if status['needs_indexing'] else 'No ✓'}")
    
    db_stats = status['database_stats']
    if db_stats.get('total_chunks', 0) > 0:
        print(f"\n📦 Database Statistics:")
        print(f"   Total chunks:       {db_stats['total_chunks']}")
        print(f"   Total tokens:       {db_stats['total_tokens']:,}")
        print(f"   Avg chunk size:     {db_stats['avg_chunk_size']:.1f} tokens")
        
        if db_stats.get('subjects_distribution'):
            print(f"\n📚 Subjects:")
            for subject, count in db_stats['subjects_distribution'].items():
                print(f"   • {subject}: {count} documents")
    
    model_info = status['model_info']
    print(f"\n🤖 Embedding Model:")
    print(f"   Model:              {model_info.get('model_name', 'N/A')}")
    print(f"   Embedding dim:      {model_info.get('embedding_dim', 'N/A')}")
    print(f"   Device:             {model_info.get('device', 'N/A')}")
    
    print("="*60 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AsylBILIM Document Indexer - Offline semantic search indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/index_documents.py                    # Index new documents
  python scripts/index_documents.py --clear            # Clear and re-index all
  python scripts/index_documents.py --status           # Show current status
  python scripts/index_documents.py --reindex DOC_ID   # Re-index specific document
        """
    )
    
    parser.add_argument(
        '--rag-dir',
        default='RAG',
        help='Directory containing JSON documents (default: RAG)'
    )
    
    parser.add_argument(
        '--db-path',
        default=settings.VECTOR_DB_PATH,
        help=f'Path to SQLite database (default: {settings.VECTOR_DB_PATH})'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing database and re-index all documents'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current indexing status'
    )
    
    parser.add_argument(
        '--reindex',
        metavar='DOC_ID',
        help='Re-index a specific document by ID'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=settings.CHUNK_SIZE,
        help=f'Chunk size in tokens (default: {settings.CHUNK_SIZE})'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=settings.CHUNK_OVERLAP,
        help=f'Overlap size in tokens (default: {settings.CHUNK_OVERLAP})'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    try:
        # Initialize indexer
        logger.info(f"Initializing indexer: {args.rag_dir} -> {args.db_path}")
        indexer = DocumentIndexer(
            rag_directory=args.rag_dir,
            db_path=args.db_path,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        
        # Handle different commands
        if args.status:
            # Show status
            status = indexer.get_indexing_status()
            print_status(status)
            
        elif args.reindex:
            # Re-index specific document
            print(f"Re-indexing document: {args.reindex}")
            success = indexer.reindex_document(args.reindex)
            
            if success:
                print(f"✓ Document '{args.reindex}' re-indexed successfully!")
            else:
                print(f"✗ Failed to re-index document '{args.reindex}'")
                sys.exit(1)
        
        else:
            # Full indexing
            if args.clear:
                print("⚠️  WARNING: This will delete all existing indexed data!")
                response = input("Continue? (yes/no): ")
                if response.lower() != 'yes':
                    print("Aborted.")
                    sys.exit(0)
            
            # Run indexing
            stats = indexer.index_all(clear_existing=args.clear)
            print_stats(stats)
            
            # Unload model to free memory
            indexer.unload_model()
            
            if not stats.get('success'):
                sys.exit(1)
        
        print("✓ Done!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()