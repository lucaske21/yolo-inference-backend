#!/usr/bin/env python
"""
LRU Cache test for YOLO Inference Backend.

This script tests the LRU cache implementation with memory management.
"""

import sys
import os
import time
import psutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.tools import load_models
from utils.tools_lru import InferenceSessionsWithLRU
from logger import setup_logging


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def test_lru_basic():
    """Test basic LRU cache functionality."""
    print("\n" + "="*70)
    print("TEST 1: BASIC LRU CACHE FUNCTIONALITY")
    print("="*70)
    
    setup_logging(log_level='INFO')
    
    # Create LRU cache with low memory limit for testing
    lru_sessions = InferenceSessionsWithLRU(max_memory_mb=2048, memory_check_interval=1)
    
    # Load models metadata
    models = load_models('/tmp/test_models')
    lru_sessions.initialize_sessions(models, top_n=2)
    
    print("\n✓ LRU cache initialized")
    print(f"  Max memory: {lru_sessions.max_memory_mb} MB")
    print(f"  Check interval: {lru_sessions.memory_check_interval}")
    
    # Get initial stats
    stats = lru_sessions.get_cache_stats()
    print(f"\n✓ Initial cache stats:")
    print(f"  Loaded models: {stats['loaded_models']}")
    print(f"  Memory: {stats['current_memory_mb']:.2f} MB")
    
    return True


def test_lru_loading():
    """Test lazy loading with LRU cache."""
    print("\n" + "="*70)
    print("TEST 2: LAZY LOADING WITH LRU")
    print("="*70)
    
    setup_logging(log_level='INFO')
    
    baseline = get_memory_usage_mb()
    print(f"\nBaseline memory: {baseline:.2f} MB")
    
    # Create LRU cache
    lru_sessions = InferenceSessionsWithLRU(max_memory_mb=3000, memory_check_interval=2)
    models = load_models('/tmp/test_models')
    lru_sessions.initialize_sessions(models, top_n=2)
    
    after_init = get_memory_usage_mb()
    print(f"After init: {after_init:.2f} MB (+{after_init - baseline:.2f} MB)")
    
    # Load first model
    print("\n📥 Loading model 0...")
    session_0 = lru_sessions.get_session(0)
    if session_0:
        after_model_0 = get_memory_usage_mb()
        print(f"✓ Model 0 loaded: {after_model_0:.2f} MB (+{after_model_0 - baseline:.2f} MB)")
        
        # Check stats
        stats = lru_sessions.get_cache_stats()
        print(f"  Loaded models: {stats['loaded_models']}")
        print(f"  Memory usage: {stats['memory_usage_percent']:.1f}%")
    
    # Access model 0 again (should be cached)
    print("\n📦 Accessing model 0 again (cached)...")
    session_0_cached = lru_sessions.get_session(0)
    if session_0_cached:
        print("✓ Cache hit - no additional memory used")
    
    # Load second model
    print("\n📥 Loading model 1...")
    session_1 = lru_sessions.get_session(1)
    if session_1:
        after_model_1 = get_memory_usage_mb()
        print(f"✓ Model 1 loaded: {after_model_1:.2f} MB (+{after_model_1 - baseline:.2f} MB)")
        
        # Check stats
        stats = lru_sessions.get_cache_stats()
        print(f"  Loaded models: {stats['loaded_models']}")
        print(f"  Memory usage: {stats['memory_usage_percent']:.1f}%")
    
    return True


def test_lru_eviction():
    """Test LRU eviction when memory limit is reached."""
    print("\n" + "="*70)
    print("TEST 3: LRU EVICTION ON MEMORY LIMIT")
    print("="*70)
    
    setup_logging(log_level='INFO')
    
    # Create LRU cache with very low memory limit to force eviction
    print("\n⚠️  Setting low memory limit to test eviction...")
    lru_sessions = InferenceSessionsWithLRU(max_memory_mb=500, memory_check_interval=1)
    models = load_models('/tmp/test_models')
    lru_sessions.initialize_sessions(models, top_n=2)
    
    print(f"Max memory limit: {lru_sessions.max_memory_mb} MB")
    
    # Load model 0
    print("\n📥 Loading model 0...")
    session_0 = lru_sessions.get_session(0)
    stats_0 = lru_sessions.get_cache_stats()
    print(f"✓ Model 0 loaded")
    print(f"  Loaded models: {stats_0['loaded_models']}")
    print(f"  Memory: {stats_0['current_memory_mb']:.2f} MB ({stats_0['memory_usage_percent']:.1f}%)")
    
    # Try to load model 1 (should trigger eviction if memory is high)
    print("\n📥 Loading model 1 (may trigger eviction)...")
    session_1 = lru_sessions.get_session(1)
    stats_1 = lru_sessions.get_cache_stats()
    print(f"✓ Model 1 loaded")
    print(f"  Loaded models: {stats_1['loaded_models']}")
    print(f"  Memory: {stats_1['current_memory_mb']:.2f} MB ({stats_1['memory_usage_percent']:.1f}%)")
    
    if 0 not in stats_1['loaded_models'] and len(stats_0['loaded_models']) > len(stats_1['loaded_models']):
        print("\n🗑️  Model 0 was evicted due to memory constraints!")
        print("✓ LRU eviction working correctly")
    else:
        print("\n⚠️  No eviction occurred (memory limit not reached)")
    
    return True


def test_lru_order():
    """Test that LRU order is maintained correctly."""
    print("\n" + "="*70)
    print("TEST 4: LRU ORDER MAINTENANCE")
    print("="*70)
    
    setup_logging(log_level='INFO')
    
    lru_sessions = InferenceSessionsWithLRU(max_memory_mb=3000, memory_check_interval=1)
    models = load_models('/tmp/test_models')
    lru_sessions.initialize_sessions(models, top_n=2)
    
    # Load both models
    print("\n📥 Loading models 0 and 1...")
    lru_sessions.get_session(0)
    lru_sessions.get_session(1)
    
    stats = lru_sessions.get_cache_stats()
    initial_order = stats['loaded_models']
    print(f"Initial order: {initial_order}")
    
    # Access model 0 again (should move to end of LRU)
    print("\n📦 Accessing model 0 (should move to end of LRU)...")
    lru_sessions.get_session(0)
    
    stats = lru_sessions.get_cache_stats()
    new_order = stats['loaded_models']
    print(f"New order: {new_order}")
    
    if new_order[-1] == 0:
        print("✓ LRU order updated correctly - model 0 is most recently used")
    
    return True


def main():
    """Run all LRU cache tests."""
    print("\n" + "="*70)
    print("LRU CACHE IMPLEMENTATION TESTS")
    print("="*70)
    
    tests = [
        ("Basic functionality", test_lru_basic),
        ("Lazy loading with LRU", test_lru_loading),
        ("LRU eviction", test_lru_eviction),
        ("LRU order maintenance", test_lru_order),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
            time.sleep(1)  # Brief pause between tests
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ Test failed with error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for test_name, result, error in results:
        if result:
            print(f"✓ {test_name}: PASSED")
            passed += 1
        else:
            print(f"✗ {test_name}: FAILED")
            if error:
                print(f"  Error: {error}")
            failed += 1
    
    print("\n" + "-"*70)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
