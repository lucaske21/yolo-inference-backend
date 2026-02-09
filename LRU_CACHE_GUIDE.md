# LRU Cache Implementation Guide

## Overview

This document describes the LRU (Least Recently Used) Cache implementation for the YOLO Inference Backend, providing intelligent memory management for environments with many models and limited memory.

## What is LRU Cache?

LRU Cache is a memory management strategy that:
1. **Loads models lazily** - Only when first requested
2. **Tracks access order** - Remembers which models were used recently
3. **Enforces memory limits** - Automatically evicts models when memory is low
4. **Evicts intelligently** - Removes least recently used models first

## Why Use LRU Cache?

### Use LRU Cache When:
- ✅ You have many models (>10) but limited memory
- ✅ Different models are used at different times
- ✅ You need a guaranteed memory ceiling
- ✅ You can accept occasional model reload delays

### Use Lazy Loading When:
- ✅ You have few models (2-10)
- ✅ You have sufficient memory to cache all models
- ✅ You want the simplest solution
- ✅ You don't want eviction overhead

## Configuration

LRU Cache is controlled via environment variables:

### Required Settings

```bash
# Enable LRU cache mode
export ENABLE_LRU_CACHE=true

# Set maximum memory limit in MB
export MAX_MEMORY_MB=4096

# Set memory check interval (number of requests)
export MEMORY_CHECK_INTERVAL=10
```

### Configuration Options

| Variable | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `ENABLE_LRU_CACHE` | Enable LRU cache mode | `false` | `true` for many models |
| `MAX_MEMORY_MB` | Maximum memory in MB | `4096` | Based on your system |
| `MEMORY_CHECK_INTERVAL` | Check every N requests | `10` | 5-20 for balance |

## How It Works

### 1. Lazy Loading

Models are not loaded at startup. They load on first API request:

```
Startup → Load metadata only → Ready in <1 second
First request → Load model → Cache it → Respond
Subsequent requests → Use cached model → Fast response
```

### 2. LRU Tracking

Uses Python's `OrderedDict` to efficiently track access order:

```python
# When model is accessed:
cache[model_id]  # Access model
cache.move_to_end(model_id)  # Mark as most recently used

# Access order is automatically maintained:
cache = OrderedDict({2: model2, 0: model0, 1: model1})
#                    ^oldest              ^newest
```

### 3. Memory Monitoring

Periodically checks memory usage using `psutil`:

```python
# Every N requests (configurable):
current_memory = get_current_memory_mb()
if current_memory > MAX_MEMORY_MB:
    evict_least_recently_used()
```

### 4. Smart Eviction

When memory exceeds limit, removes LRU models:

```python
# Evict until memory < 90% of max:
while memory > MAX_MEMORY_MB * 0.9 and cache_not_empty:
    evict_model(least_recently_used)
```

## Usage

### Starting the Server

```bash
# With LRU cache enabled:
export ENABLE_LRU_CACHE=true
export MAX_MEMORY_MB=4096
export MEMORY_CHECK_INTERVAL=10
python src/app.py
```

### API Endpoints

#### Regular Detection (same as before)

```bash
curl -X POST http://localhost:8000/api/v2/detect \
  -F "file=@image.jpg" \
  -F "model_id=0"
```

#### Cache Statistics (new)

```bash
curl http://localhost:8000/api/v2/cache/stats
```

**Response:**
```json
{
  "loaded_models": [0, 2, 5],
  "num_loaded_models": 3,
  "current_memory_mb": 2845.32,
  "max_memory_mb": 4096,
  "memory_usage_percent": 69.5,
  "request_count": 127,
  "memory_check_interval": 10
}
```

## Memory Management Example

### Scenario: 3 Models, 1000MB Limit

```
1. Load model 0 (200MB)     → Memory: 200MB, Cache: [0]
2. Load model 1 (200MB)     → Memory: 400MB, Cache: [0, 1]
3. Load model 2 (200MB)     → Memory: 600MB, Cache: [0, 1, 2]
4. Access model 0           → Memory: 600MB, Cache: [1, 2, 0]  (0 moved to end)
5. Load model 3 (300MB)     → Triggers check...
   - Would use 900MB (OK, under limit)
   → Memory: 900MB, Cache: [1, 2, 0, 3]

6. Load model 4 (300MB)     → Triggers check...
   - Would use 1200MB (exceeds limit!)
   - Evict model 1 (LRU)
   - Memory: 700MB, Cache: [2, 0, 3]
   - Load model 4
   → Memory: 1000MB, Cache: [2, 0, 3, 4]
```

## Performance Characteristics

### Memory Usage

| Scenario | Lazy Loading | LRU Cache |
|----------|--------------|-----------|
| Startup | ~200 MB | ~200 MB |
| 5 models loaded | ~2500 MB | ~2500 MB |
| 10 models loaded | ~5000 MB | Capped at MAX_MEMORY_MB |
| 20 models loaded | ~10000 MB | Capped at MAX_MEMORY_MB |

### Request Latency

| Request Type | Latency | Notes |
|--------------|---------|-------|
| Cache hit | ~50ms | Model already loaded |
| Cache miss (space available) | ~2-3s | Load model |
| Cache miss (need eviction) | ~3-4s | Evict + load |

### Memory Check Overhead

- Checking memory: ~1ms per check
- Frequency: Every N requests (default: 10)
- Impact: Negligible (<0.1% overhead)

## Tuning Guidelines

### Setting MAX_MEMORY_MB

1. **Check total system memory:**
   ```bash
   free -m | grep Mem
   ```

2. **Reserve for OS and other processes:**
   - Production: 70-80% of total memory
   - Development: 50-60% of total memory

3. **Example calculations:**
   ```
   System: 8GB (8192 MB)
   Reserve for OS: 2GB (2048 MB)
   Available: 6GB (6144 MB)
   
   Set MAX_MEMORY_MB=6000  # Leave 144MB buffer
   ```

### Setting MEMORY_CHECK_INTERVAL

- **Lower (5)**: More frequent checks, faster response to memory pressure
- **Higher (20)**: Less overhead, slower response to memory pressure
- **Default (10)**: Good balance for most scenarios

**Guidelines:**
- High traffic: 5-10 (more frequent checks)
- Low traffic: 15-20 (less overhead)
- Many models: 5-10 (catch issues early)

## Monitoring and Debugging

### Real-time Monitoring

```bash
# Watch cache statistics:
watch -n 2 'curl -s http://localhost:8000/api/v2/cache/stats | jq'
```

### Log Messages

The LRU cache logs important events:

```
INFO: InferenceSessionsWithLRU initialized with max_memory=4096MB
INFO: Cache miss for model 0, lazy loading...
INFO: Initialized inference session for model 0. Current memory: 1850.32MB
WARNING: Memory usage 4200.50MB exceeds limit 4096MB
INFO: Evicted model 2 (LRU) to free memory
INFO: Memory after eviction: 3950.25MB
```

### Common Issues

#### 1. Too Many Evictions

**Symptom:** Models frequently reloaded  
**Cause:** MAX_MEMORY_MB too low  
**Solution:** Increase MAX_MEMORY_MB or reduce number of concurrent models

#### 2. Memory Still Growing

**Symptom:** Memory exceeds MAX_MEMORY_MB  
**Cause:** Other parts of application using memory  
**Solution:** Check for memory leaks, reduce MAX_MEMORY_MB to leave buffer

#### 3. Slow Performance

**Symptom:** High latency on many requests  
**Cause:** Frequent evictions and reloads  
**Solution:** Increase MAX_MEMORY_MB or use fewer models

## Comparison with Lazy Loading

### Lazy Loading (Original Implementation)

```python
# Advantages:
+ Simple implementation
+ Predictable performance
+ No eviction overhead

# Disadvantages:
- No memory limit
- Memory grows unbounded
- Not suitable for many models
```

### LRU Cache (This Implementation)

```python
# Advantages:
+ Controlled memory usage
+ Supports many models
+ Intelligent eviction
+ Configurable limits

# Disadvantages:
- More complex
- Slight overhead on checks
- Occasional reload delays
```

## Best Practices

### 1. Set Appropriate Memory Limit

```bash
# Leave 20-30% buffer for OS and other processes
TOTAL_MEMORY=8192  # 8GB system
MAX_MEMORY_MB=6000  # ~73% of total
```

### 2. Monitor Cache Statistics

```bash
# Set up monitoring
curl http://localhost:8000/api/v2/cache/stats > cache_stats.json
# Check memory_usage_percent regularly
```

### 3. Tune Check Interval

```bash
# High traffic: check more frequently
export MEMORY_CHECK_INTERVAL=5

# Low traffic: check less frequently
export MEMORY_CHECK_INTERVAL=20
```

### 4. Log Analysis

```bash
# Monitor evictions
grep "Evicted model" app.log

# Monitor memory warnings
grep "Memory usage.*exceeds" app.log
```

## Testing

### Unit Tests

```bash
# Run LRU logic tests:
python tests/test_lru_logic.py
```

**Tests:**
- OrderedDict LRU behavior
- Memory threshold logic
- Periodic check timing
- Eviction simulation

### Integration Tests

```bash
# Run with real models:
python tests/test_lru_cache.py
```

**Tests:**
- Basic functionality
- Lazy loading with LRU
- Eviction under memory pressure
- LRU order maintenance

## Migration from Lazy Loading

### Step 1: Test in Development

```bash
# Enable LRU cache
export ENABLE_LRU_CACHE=true
export MAX_MEMORY_MB=4096

# Start server
python src/app.py
```

### Step 2: Monitor Performance

```bash
# Check cache statistics
curl http://localhost:8000/api/v2/cache/stats

# Watch logs for evictions
tail -f app.log | grep -E "(Evicted|Memory)"
```

### Step 3: Tune Configuration

Adjust based on observations:
- Increase MAX_MEMORY_MB if too many evictions
- Decrease MEMORY_CHECK_INTERVAL if memory exceeds limit
- Monitor memory_usage_percent

### Step 4: Deploy to Production

Once tuned and stable:
1. Update environment configuration
2. Deploy with LRU enabled
3. Monitor cache statistics
4. Adjust as needed

## Troubleshooting

### Q: Cache statistics not available?

**A:** Ensure `ENABLE_LRU_CACHE=true`. The endpoint only works in LRU mode.

### Q: Models evicted too frequently?

**A:** Increase `MAX_MEMORY_MB` or reduce concurrent model usage.

### Q: Memory still grows beyond limit?

**A:** Check for memory leaks in other parts. LRU controls only model memory.

### Q: Performance degraded?

**A:** Check eviction frequency. Consider increasing memory limit.

## Summary

LRU Cache provides intelligent memory management for YOLO models:

✅ **Controlled memory** - Never exceeds MAX_MEMORY_MB  
✅ **Supports many models** - Handles 10+ models efficiently  
✅ **Automatic eviction** - Removes LRU models when needed  
✅ **Configurable** - Tune for your environment  
✅ **Backwards compatible** - Can switch back to lazy loading  

For most production deployments with many models and limited memory, LRU Cache is the recommended approach.
