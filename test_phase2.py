#!/usr/bin/env python3
"""Quick test script for Phase 2 implementations."""

import asyncio
import tempfile
from pathlib import Path

import numpy as np

from ai_unit_test.core.factories.index_factory import IndexOrganizerFactory
from ai_unit_test.core.factories.llm_factory import LLMConnectorFactory
from ai_unit_test.core.interfaces.llm_connector import LLMRequest


async def test_implementations():
    """Test all Phase 2 implementations."""
    print("=== Phase 2 Implementation Test ===")

    # Test LLM Factories
    print("\n1. Testing LLM Connector Factory:")
    available_llms = LLMConnectorFactory.get_available_connectors()
    print(f"Available LLM connectors: {available_llms}")

    # Test Mock Connector
    if "mock" in available_llms:
        print("\n2. Testing Mock Connector:")
        mock_connector = LLMConnectorFactory.create_connector("mock")
        await mock_connector.initialize()

        request = LLMRequest(
            system_message="You are helpful.",
            user_message="Write a test function",
            model="mock-model",
            temperature=0.7,
            max_tokens=100,
        )

        response = await mock_connector.generate_response(request)
        print(f"Mock response: {response.content[:100]}...")
        print(f"Health check: {await mock_connector.health_check()}")

    # Test Index Organizer Factory
    print("\n3. Testing Index Organizer Factory:")
    available_organizers = IndexOrganizerFactory.get_available_organizers()
    print(f"Available organizers: {available_organizers}")

    # Test Memory Organizer
    if "memory" in available_organizers:
        print("\n4. Testing Memory Organizer:")
        memory_organizer = IndexOrganizerFactory.create_organizer("memory")

        # Create sample data
        embeddings = np.random.random((10, 128)).astype(np.float32)
        metadata = [{"id": i, "text": f"sample text {i}"} for i in range(10)]

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test_index"

            # Create index
            index_info = await memory_organizer.create_index(
                embeddings=embeddings, metadata=metadata, index_path=index_path, model_name="test-model"
            )
            print(f"Created index with {index_info.total_documents} documents")

            # Test search
            query = np.random.random((1, 128)).astype(np.float32)
            results = await memory_organizer.search(query, k=3, threshold=0.0)
            print(f"Search returned {len(results)} results")

            # Test stats
            stats = await memory_organizer.get_stats()
            print(f"Index stats - Documents: {stats.total_documents}, Memory: {stats.memory_usage_mb:.2f}MB")

    print("\n✅ Phase 2 implementation test completed!")


if __name__ == "__main__":
    asyncio.run(test_implementations())
