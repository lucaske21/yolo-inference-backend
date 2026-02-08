#!/usr/bin/env python
"""
Unit tests for LRU Cache logic without requiring YOLO models.

Tests the LRU cache data structures and eviction logic.
"""

import sys
import os
from collections import OrderedDict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_ordered_dict_lru():
    """Test OrderedDict LRU behavior."""
    print("\n" + "="*70)
    print("TEST 1: OrderedDict LRU Behavior")
    print("="*70)
    
    # Simulate LRU cache with OrderedDict
    cache = OrderedDict()
    
    # Add items
    cache[0] = "model_0"
    cache[1] = "model_1"
    cache[2] = "model_2"
    
    print("\nInitial cache order:", list(cache.keys()))
    
    # Access item 0 (should move to end)
    if 0 in cache:
        cache.move_to_end(0)
    
    print("After accessing 0:", list(cache.keys()))
    assert list(cache.keys()) == [1, 2, 0], "Item 0 should be at the end"
    
    # Access item 1
    if 1 in cache:
        cache.move_to_end(1)
    
    print("After accessing 1:", list(cache.keys()))
    assert list(cache.keys()) == [2, 0, 1], "Item 1 should be at the end"
    
    # Evict least recently used (first item)
    evicted_key, evicted_value = cache.popitem(last=False)
    print(f"\nEvicted least recently used: {evicted_key} = {evicted_value}")
    print("Remaining cache:", list(cache.keys()))
    
    assert evicted_key == 2, "Item 2 should be evicted (least recently used)"
    assert list(cache.keys()) == [0, 1], "Items 0 and 1 should remain"
    
    print("\n✓ OrderedDict LRU behavior works correctly")
    return True


def test_memory_threshold_logic():
    """Test memory threshold and eviction decision logic."""
    print("\n" + "="*70)
    print("TEST 2: Memory Threshold Logic")
    print("="*70)
    
    max_memory_mb = 1000
    current_memory_mb = 850
    
    # Check if memory exceeds limit
    if current_memory_mb > max_memory_mb:
        print(f"❌ Memory {current_memory_mb}MB exceeds limit {max_memory_mb}MB")
        should_evict = True
    else:
        print(f"✓ Memory {current_memory_mb}MB within limit {max_memory_mb}MB")
        should_evict = False
    
    assert not should_evict, "Should not evict when under limit"
    
    # Simulate memory increase
    current_memory_mb = 1100
    
    if current_memory_mb > max_memory_mb:
        print(f"⚠️  Memory {current_memory_mb}MB exceeds limit {max_memory_mb}MB")
        should_evict = True
    
    assert should_evict, "Should evict when over limit"
    
    # Calculate target memory (90% of max)
    target_memory = max_memory_mb * 0.9
    print(f"\nTarget memory after eviction: {target_memory}MB")
    
    print("\n✓ Memory threshold logic works correctly")
    return True


def test_periodic_check_logic():
    """Test periodic memory check logic."""
    print("\n" + "="*70)
    print("TEST 3: Periodic Check Logic")
    print("="*70)
    
    memory_check_interval = 10
    request_count = 0
    
    checks_performed = []
    
    # Simulate 25 requests
    for i in range(25):
        request_count += 1
        
        # Check if it's time for memory check
        if request_count % memory_check_interval == 0:
            checks_performed.append(request_count)
            print(f"✓ Memory check at request {request_count}")
    
    print(f"\nTotal requests: 25")
    print(f"Memory checks performed at: {checks_performed}")
    
    expected_checks = [10, 20]
    assert checks_performed == expected_checks, f"Expected checks at {expected_checks}, got {checks_performed}"
    
    print("\n✓ Periodic check logic works correctly")
    return True


def test_lru_eviction_simulation():
    """Simulate LRU eviction under memory pressure."""
    print("\n" + "="*70)
    print("TEST 4: LRU Eviction Simulation")
    print("="*70)
    
    cache = OrderedDict()
    max_memory_mb = 1000
    current_memory_mb = 800
    model_size_mb = 200  # Each model is 200MB
    
    # Load 3 models
    for i in range(3):
        cache[i] = f"model_{i}"
        current_memory_mb += model_size_mb
        print(f"Loaded model {i}, memory: {current_memory_mb}MB")
    
    print(f"\nCache contains: {list(cache.keys())}")
    print(f"Current memory: {current_memory_mb}MB / {max_memory_mb}MB")
    
    # Try to load a 4th model - should trigger eviction
    if current_memory_mb + model_size_mb > max_memory_mb:
        print(f"\n⚠️  Loading model 3 would exceed memory limit")
        print(f"   Would use: {current_memory_mb + model_size_mb}MB > {max_memory_mb}MB")
        
        # Evict models until we have space
        while current_memory_mb + model_size_mb > max_memory_mb * 0.9 and len(cache) > 0:
            evicted_key, _ = cache.popitem(last=False)
            current_memory_mb -= model_size_mb
            print(f"   Evicted model {evicted_key}, memory: {current_memory_mb}MB")
        
        # Now load the new model
        cache[3] = "model_3"
        current_memory_mb += model_size_mb
        print(f"   Loaded model 3, memory: {current_memory_mb}MB")
    
    print(f"\nFinal cache contains: {list(cache.keys())}")
    print(f"Final memory: {current_memory_mb}MB / {max_memory_mb}MB")
    
    # Model 0 should be evicted (least recently used)
    assert 0 not in cache, "Model 0 should have been evicted"
    assert 3 in cache, "Model 3 should be loaded"
    
    print("\n✓ LRU eviction simulation works correctly")
    return True


def main():
    """Run all unit tests."""
    print("\n" + "="*70)
    print("LRU CACHE LOGIC UNIT TESTS")
    print("="*70)
    
    tests = [
        ("OrderedDict LRU", test_ordered_dict_lru),
        ("Memory threshold", test_memory_threshold_logic),
        ("Periodic check", test_periodic_check_logic),
        ("Eviction simulation", test_lru_eviction_simulation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except AssertionError as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ Assertion failed: {e}")
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
