# REDUNDANCY ANALYSIS PROTOCOL - EXTREME CAUTION REQUIRED

## CRITICAL SAFETY RULES

1. **NEVER DELETE WITHOUT COMPLETE ANALYSIS** - Read every single line of both files
2. **FEATURE PRESERVATION MANDATORY** - Ensure 100% feature retention in kept file
3. **LINE-BY-LINE COMPARISON** - Compare every line, function, class, and comment
4. **ARCHIVE BEFORE REMOVAL** - Always archive the redundant file with timestamp
5. **VERIFICATION REQUIRED** - Test functionality after any consolidation
6. **DOCUMENTATION MANDATORY** - Document every decision with evidence

## ANALYSIS PROCESS

### Phase 1: File Identification
- Identify potentially redundant files by name patterns
- Group by functionality (analytics, monitoring, testing, etc.)
- Create candidate pairs for detailed analysis

### Phase 2: Complete File Reading
For each candidate pair:
1. Read File A completely (every line)
2. Read File B completely (every line)
3. Create detailed feature inventory for each
4. Document unique features in each file
5. Identify true overlaps vs. similar names

### Phase 3: Feature Mapping
- Map every function/class/feature between files
- Identify which file has the most complete implementation
- Document any unique features that must be preserved
- Note any configuration differences or parameters

### Phase 4: Safety Verification
- Confirm the "kept" file contains ALL features from "removed" file
- Verify no unique logic is lost
- Check for different parameter handling or edge cases
- Validate no tests would break

### Phase 5: Archive and Remove
Only after 100% verification:
1. Create timestamped archive copy
2. Update archive manifest
3. Remove redundant file
4. Test affected systems
5. Document action in progress log

## EXAMPLE WORKFLOW

```
Candidate: analytics_aggregator.py vs analytics_collector.py

1. Read analytics_aggregator.py (all 234 lines)
   - Functions: aggregate_metrics(), batch_process(), store_results()
   - Classes: MetricAggregator, BatchProcessor
   - Unique features: Time-based aggregation, Custom storage format

2. Read analytics_collector.py (all 187 lines)  
   - Functions: collect_data(), process_batch(), save_metrics()
   - Classes: DataCollector, MetricProcessor
   - Unique features: Real-time collection, Different data sources

3. Analysis: NOT REDUNDANT - Different purposes
   - aggregator: Post-processing of collected data
   - collector: Initial data gathering
   - Decision: KEEP BOTH
```

## NO SHORTCUTS ALLOWED

- Never assume files are redundant based on names alone
- Never skip reading any portion of files
- Never remove files without complete verification
- Never trust previous analysis - verify everything
- Always err on the side of caution - keep both if uncertain

This protocol MUST be followed for every single file analysis.