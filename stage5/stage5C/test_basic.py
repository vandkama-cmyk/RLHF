#!/usr/bin/env python3
"""
Basic test script for ClassifLLM v3
==================================

Tests the system without requiring real LLM API calls.
Uses synthetic feedback generation for testing.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")

    try:
        from stage5.stage5C.llm_feedback import LLMFeedbackGenerator, LLMConfig, create_default_config
        from stage5.stage5C.integrated_system import EnhancedLLMTrainingPipeline, EnhancedCodeQualityDataset
        from stage5.stage5C.config import get_example_config
        print("[OK] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    print("Testing configuration creation...")

    try:
        from stage5.stage5C.config import get_example_config, create_training_config

        # Test synthetic config creation
        config = create_training_config(llm_config=None, preset='research')
        assert config['llm_config'] is None
        assert 'batch_size' in config  # Check that config has training parameters

        # Test local config
        from stage5.stage5C.config import get_local_config
        llm_config = get_local_config("microsoft/DialoGPT-small")
        config = create_training_config(llm_config=llm_config, preset='fast')
        assert config['llm_config'].provider == 'local'
        assert 'microsoft/DialoGPT-small' in config['llm_config'].model_name

        print("[OK] Configuration creation successful")
        return True
    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False

def test_synthetic_feedback():
    """Test synthetic feedback generation."""
    print("Testing synthetic feedback generation...")

    try:
        from stage5.stage5C.llm_feedback import LLMFeedbackGenerator, LLMConfig

        # Create a mock provider that doesn't require API
        class MockProvider:
            async def generate_feedback(self, question, answer, context=None):
                from stage5.stage5C.llm_feedback import CodeQualityFeedback
                # Return mock feedback
                return CodeQualityFeedback(
                    consistent=0.8,
                    correct=0.9,
                    useful=0.7,
                    explanation="Mock feedback for testing",
                    confidence=0.8
                )

        # Create generator with mock provider
        config = LLMConfig(provider="mock", model_name="test")
        generator = LLMFeedbackGenerator.__new__(LLMFeedbackGenerator)
        generator.config = config
        generator.provider = MockProvider()
        generator.cache = {}

        # Test sample
        samples = [{
            'question': 'How to sort a list?',
            'answer': 'sorted_list = sorted(my_list)',
            'metadata': {'source': 'test'}
        }]

        # Generate feedback
        async def run_test():
            results = await generator.generate_feedback_batch(samples, show_progress=False)
            assert len(results) == 1
            assert 'feedback' in results[0]
            assert 'labels' in results[0]
            print("[OK] Synthetic feedback generation successful")
            return True

        return asyncio.run(run_test())

    except Exception as e:
        print(f"[FAIL] Synthetic feedback test failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation."""
    print("Testing dataset creation...")

    try:
        from stage5.stage5C.integrated_system import EnhancedCodeQualityDataset, generate_base_samples

        # Generate some samples
        samples = generate_base_samples(50)
        assert len(samples) == 50

        # Create dataset
        dataset = EnhancedCodeQualityDataset(samples)
        assert len(dataset) == 50

        # Test item retrieval
        item = dataset[0]
        assert 'question' in item
        assert 'answer' in item
        assert 'labels' in item

        print("[OK] Dataset creation successful")
        return True
    except Exception as e:
        print(f"[FAIL] Dataset test failed: {e}")
        return False

def test_training_pipeline_init():
    """Test training pipeline initialization."""
    print("Testing training pipeline initialization...")

    try:
        from stage5.stage5C.integrated_system import EnhancedLLMTrainingPipeline

        config = {
            'device': 'cpu',  # Use CPU for testing
            'model_type': 'codebert',
            'learning_rate': 2e-5,
            'dropout': 0.3,
            'freeze_layers': 8,  # Freeze more for faster testing
            'batch_size': 8,
            'epochs': 1,
            'patience': 3,
            'use_amp': False,  # Disable AMP for CPU
            'use_cross_attention': False,  # Disable for faster init
            'use_contrastive': False,  # Disable for faster init
            'max_length': 128,
        }

        pipeline = EnhancedLLMTrainingPipeline(config)
        assert pipeline.model is not None
        assert pipeline.device == 'cpu'

        print("[OK] Training pipeline initialization successful")
        return True
    except Exception as e:
        print(f"[FAIL] Training pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("  ClassifLLM v3 - Basic Functionality Tests")
    print("=" * 60)
    print()

    tests = [
        test_imports,
        test_config_creation,
        test_synthetic_feedback,
        test_dataset_creation,
        test_training_pipeline_init,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} crashed: {e}")
            print()

    print("=" * 60)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("[SUCCESS] All tests passed! ClassifLLM v3 is ready to use.")
        print()
        print("Next steps:")
        print("1. Set up your LLM API key (if using OpenAI/Anthropic)")
        print("2. Run: python clasifLLM/v3/train.py --help")
        print("3. Start training: python clasifLLM/v3/train.py [options]")
    else:
        print("[ERROR] Some tests failed. Please check the errors above.")

    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
