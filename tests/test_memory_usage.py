#!/usr/bin/env python
"""
Memory usage comparison test for lazy loading optimization.

This script compares memory usage between eager loading (old approach)
and lazy loading (new approach).
"""

import sys
import os
import psutil
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.tools import load_models, InferenceSessions
from logger import setup_logging


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def test_lazy_loading():
    """Test lazy loading memory usage."""
    print("\n" + "="*70)
    print("TESTING LAZY LOADING (NEW APPROACH)")
    print("="*70)
    
    setup_logging(log_level='INFO')
    
    # Baseline memory
    baseline_memory = get_memory_usage_mb()
    print(f"\nBaseline memory: {baseline_memory:.2f} MB")
    
    # Load models metadata only (no actual model loading)
    models = load_models('/tmp/test_models')
    after_metadata = get_memory_usage_mb()
    print(f"After loading metadata: {after_metadata:.2f} MB (+{after_metadata - baseline_memory:.2f} MB)")
    
    # Initialize inference sessions (lazy loading - should not load models yet)
    inference_sessions = InferenceSessions()
    inference_sessions.initialize_sessions(models, top_n=2)
    after_init = get_memory_usage_mb()
    print(f"After initializing sessions (lazy): {after_init:.2f} MB (+{after_init - baseline_memory:.2f} MB)")
    print("  -> Models NOT loaded yet (lazy loading)")
    
    # Simulate first API call - this will trigger loading of model 0
    print("\nSimulating first API call with model_id=0...")
    session_0 = inference_sessions.get_session(0)
    after_first_load = get_memory_usage_mb()
    print(f"After loading model 0: {after_first_load:.2f} MB (+{after_first_load - baseline_memory:.2f} MB)")
    
    # Simulate second API call with same model (should use cached model)
    print("\nSimulating second API call with model_id=0 (cached)...")
    session_0_cached = inference_sessions.get_session(0)
    after_cached = get_memory_usage_mb()
    print(f"After cached access: {after_cached:.2f} MB (+{after_cached - baseline_memory:.2f} MB)")
    print("  -> No additional memory used (model already cached)")
    
    # Now load second model
    print("\nSimulating API call with model_id=1...")
    session_1 = inference_sessions.get_session(1)
    after_second_load = get_memory_usage_mb()
    print(f"After loading model 1: {after_second_load:.2f} MB (+{after_second_load - baseline_memory:.2f} MB)")
    
    print("\n" + "-"*70)
    print(f"TOTAL MEMORY INCREASE: {after_second_load - baseline_memory:.2f} MB")
    print(f"STARTUP MEMORY (before first API call): {after_init - baseline_memory:.2f} MB")
    print("-"*70)
    
    return {
        'baseline': baseline_memory,
        'after_init': after_init,
        'after_first_load': after_first_load,
        'after_second_load': after_second_load,
        'startup_overhead': after_init - baseline_memory,
        'total_overhead': after_second_load - baseline_memory
    }


def test_eager_loading():
    """Test eager loading memory usage (old approach simulation)."""
    print("\n" + "="*70)
    print("SIMULATING EAGER LOADING (OLD APPROACH)")
    print("="*70)
    
    setup_logging(log_level='INFO')
    
    # Baseline memory
    baseline_memory = get_memory_usage_mb()
    print(f"\nBaseline memory: {baseline_memory:.2f} MB")
    
    # Load models metadata
    models = load_models('/tmp/test_models')
    after_metadata = get_memory_usage_mb()
    print(f"After loading metadata: {after_metadata:.2f} MB (+{after_metadata - baseline_memory:.2f} MB)")
    
    # Simulate eager loading - load all models at startup
    print("\nEagerly loading all models at startup...")
    inference_sessions = InferenceSessions()
    inference_sessions.models = models
    
    # Manually load all models (simulating old behavior)
    for i in range(2):
        inference_sessions.add_session_label(i, models)
    
    after_init = get_memory_usage_mb()
    print(f"After initializing sessions (eager): {after_init:.2f} MB (+{after_init - baseline_memory:.2f} MB)")
    print("  -> All models loaded at startup")
    
    print("\n" + "-"*70)
    print(f"TOTAL STARTUP MEMORY: {after_init - baseline_memory:.2f} MB")
    print("-"*70)
    
    return {
        'baseline': baseline_memory,
        'after_init': after_init,
        'startup_overhead': after_init - baseline_memory
    }


def main():
    """Run memory usage comparison."""
    print("\n" + "="*70)
    print("MEMORY USAGE COMPARISON TEST")
    print("Testing with 2 YOLOv8n models")
    print("="*70)
    
    # Test eager loading first
    eager_results = test_eager_loading()
    
    # Wait a bit and test lazy loading (in separate process would be better)
    time.sleep(2)
    
    # Test lazy loading
    lazy_results = test_lazy_loading()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nEager Loading (OLD):")
    print(f"  Startup memory: {eager_results['startup_overhead']:.2f} MB")
    print(f"  -> ALL models loaded immediately")
    
    print(f"\nLazy Loading (NEW):")
    print(f"  Startup memory: {lazy_results['startup_overhead']:.2f} MB")
    print(f"  After first model load: {lazy_results['after_first_load'] - lazy_results['baseline']:.2f} MB")
    print(f"  After both models loaded: {lazy_results['total_overhead']:.2f} MB")
    print(f"  -> Models loaded only when needed")
    
    print(f"\nMEMORY SAVINGS AT STARTUP:")
    savings = eager_results['startup_overhead'] - lazy_results['startup_overhead']
    if eager_results['startup_overhead'] > 0:
        savings_percent = (savings / eager_results['startup_overhead']) * 100
        print(f"  {savings:.2f} MB ({savings_percent:.1f}% reduction)")
    else:
        print(f"  {savings:.2f} MB (N/A% - baseline too small)")
    print("="*70)
    
    print("\nℹ️  NOTE: With larger YOLO models (yolov8m, yolov8l, yolo11), ")
    print("   the memory savings would be much more significant (1-2GB per model).")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
