#!/usr/bin/env python
"""
Visual demonstration of memory optimization.
Shows a clear before/after comparison.
"""

def print_memory_bar(label, memory_mb, max_memory=3000, width=50):
    """Print a visual bar showing memory usage."""
    if memory_mb < 0:
        memory_mb = 0
    bar_length = int((memory_mb / max_memory) * width)
    bar = '█' * bar_length + '░' * (width - bar_length)
    percentage = (memory_mb / max_memory) * 100
    print(f"{label:25} [{bar}] {memory_mb:,.0f} MB ({percentage:.1f}%)")


def main():
    print("\n" + "="*80)
    print(" " * 20 + "YOLO Inference Backend - Memory Optimization")
    print("="*80)
    
    print("\n📊 MEMORY USAGE COMPARISON\n")
    
    # Before optimization
    print("🔴 BEFORE (Eager Loading - Old Approach)")
    print("-" * 80)
    baseline = 200  # Base Python + FastAPI
    model1 = 1400   # First YOLO model
    model2 = 1400   # Second YOLO model
    
    print_memory_bar("Startup (baseline)", baseline)
    print_memory_bar("+ Model 0 loaded", baseline + model1)
    print_memory_bar("+ Model 1 loaded", baseline + model1 + model2)
    
    total_before = baseline + model1 + model2
    print("\n" + "─" * 80)
    print(f"{'TOTAL STARTUP MEMORY:':25} {total_before:,.0f} MB")
    print("─" * 80)
    print("⚠️  Both models loaded immediately at startup, even if never used")
    print("⚠️  Long startup time (10-30 seconds)")
    print()
    
    # After optimization
    print("\n🟢 AFTER (Lazy Loading - New Approach)")
    print("-" * 80)
    
    print_memory_bar("Startup (baseline)", baseline)
    print("   ✓ Models NOT loaded at startup - Ready immediately!")
    
    print("\n   [First API request with model_id=0]")
    print_memory_bar("+ Model 0 loaded", baseline + model1)
    print("   ✓ Model 0 loaded on-demand")
    
    print("\n   [Second API request with model_id=0]")
    print_memory_bar("+ Model 0 (cached)", baseline + model1)
    print("   ✓ Using cached model - No additional memory")
    
    print("\n   [First API request with model_id=1]")
    print_memory_bar("+ Model 1 loaded", baseline + model1 + model2)
    print("   ✓ Model 1 loaded on-demand")
    
    print("\n" + "─" * 80)
    print(f"{'STARTUP MEMORY:':25} {baseline:,.0f} MB")
    print(f"{'TOTAL (after all models used):':25} {baseline + model1 + model2:,.0f} MB")
    print("─" * 80)
    print("✅ Fast startup (< 1 second)")
    print("✅ Models loaded only when needed")
    print("✅ Memory grows gradually as models are used")
    print()
    
    # Savings summary
    print("\n💰 MEMORY SAVINGS")
    print("="*80)
    savings = total_before - baseline
    savings_percent = (savings / total_before) * 100
    
    print(f"\nStartup Memory Reduction:")
    print(f"  Before: {total_before:,} MB")
    print(f"  After:  {baseline:,} MB")
    print(f"  Saved:  {savings:,} MB ({savings_percent:.1f}% reduction)")
    
    print(f"\n⚡ Startup Time Improvement:")
    print(f"  Before: 10-30 seconds")
    print(f"  After:  < 1 second")
    print(f"  Improvement: 10-30x faster")
    
    print("\n" + "="*80)
    print("\n📝 Note: These are estimates for medium-sized YOLO models (YOLOv8m/YOLO11)")
    print("         Actual values may vary based on model size and system configuration")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
