#!/usr/bin/env python3
"""
Comprehensive RAG integration verification
"""
import os
import sys
sys.path.append('.')

def test_document_retrieval():
    """Test if document retrieval actually works"""
    print("=== RAG Integration Verification ===")

    try:
        from src.services.document_loader import DocumentLoader

        # Load documents
        doc_loader = DocumentLoader()
        documents = doc_loader.load_documents()

        print(f"1. Documents loaded: {len(documents)}")

        # Test different search queries
        test_queries = [
            "python",
            "математика",
            "функция",
            "алгебра",
            "рим"
        ]

        for query in test_queries:
            # Simulate the retrieval logic from AI service
            relevant_docs = []
            query_lower = query.lower()

            for doc in documents:
                relevance_score = 0
                content_lower = doc.content.lower()
                metadata = doc.metadata

                # Check matches
                if (query_lower in content_lower or
                    any(query_lower in str(value).lower() for value in metadata.values())):

                    if query_lower in content_lower:
                        relevance_score += 10

                    if 'subject' in metadata and query_lower in metadata['subject'].lower():
                        relevance_score += 5

                    if 'topic' in metadata and query_lower in metadata['topic'].lower():
                        relevance_score += 5

                    if relevance_score > 0:
                        relevant_docs.append((doc, relevance_score))

            # Sort and show results
            relevant_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = relevant_docs[:3]

            print(f"\n2. Query '{query}': Found {len(top_docs)} relevant documents")
            for i, (doc, score) in enumerate(top_docs, 1):
                print(f"   {i}. {doc.id} (score: {score})")
                print(f"      Subject: {doc.metadata.get('subject', 'N/A')}")
                print(f"      Topic: {doc.metadata.get('topic', 'N/A')}")
                print(f"      Content length: {len(doc.content)} chars")

        # Test AI service integration
        print("\n3. Testing AI Service integration...")
        try:
            class MockCache:
                def get_user_history(self, uid): return []
                def get_cached_response(self, prompt): return None

            from src.services.ai import AIService

            mock_cache = MockCache()
            ai_service = AIService(
                api_key="test_key",
                model_name="gemini-pro",
                cache_service=mock_cache,
                document_loader=doc_loader
            )

            # Test the actual retrieval method
            test_query = "python функция"
            context = ai_service._retrieve_relevant_documents(test_query)

            print(f"   Query: '{test_query}'")
            if context:
                print(f"   Retrieved context length: {len(context)} characters")
                print("   Context preview:")
                print(f"   {str(context)[:200]}...")
            else:
                print("   No context retrieved")

        except Exception as e:
            print(f"   AI Service test failed: {e}")

        print("\n=== Verification Complete ===")

    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_document_retrieval()