# 🔧 UTILITY EXTRACTION PLAN
**Agent**: Agent C (Latin Swarm)  
**Timestamp**: 2025-08-23 02:15:00 UTC  
**Phase**: Phase 0 Modularization Blitz I - Hour 2  
**Focus**: Common utility pattern identification and consolidation strategy  

## 🔍 UTILITY PATTERN ANALYSIS

### **IDENTIFIED UTILITY PATTERNS:**

#### **1. Modularization Utilities** (`automate_modularization.py`)
- **Pattern**: `_create_shared_utils()` method
- **Function**: Creates shared utility modules during modularization
- **Consolidation Opportunity**: This IS the utility consolidation system
- **Action**: LEVERAGE rather than duplicate

#### **2. Intelligence Linkage Utilities** (`web/enhanced_intelligence_linkage.py`)
- **Patterns Found**: 
  - `def helper_*` functions
  - `def util_*` functions  
  - `def format_*` functions
  - `def convert_*` functions
- **Consolidation Potential**: HIGH - multiple utility function patterns

#### **3. Documentation Generator Utilities** (`core/analysis/predictive_intelligence/documentation_generator.py`)
- **Pattern**: Utility class naming conventions
- **Function**: Helper class identification and description generation
- **Consolidation**: Can be extracted to shared utility module

## 🏗️ CONSOLIDATION STRATEGY

### **PHASE 1: LEVERAGE EXISTING AUTOMATION**
**Use `ModularizationAutomator._create_shared_utils()` method:**
- This method already creates `shared_utils.py` files
- Contains common functionality extraction logic
- Follows established patterns for utility consolidation

### **PHASE 2: IDENTIFY EXTRACTION TARGETS**
**Primary Candidates:**
1. **Helper Functions**: Functions matching `helper_*` pattern
2. **Utility Functions**: Functions matching `util_*` pattern
3. **Format Functions**: Functions matching `format_*` pattern
4. **Convert Functions**: Functions matching `convert_*` pattern

### **PHASE 3: CONSOLIDATION LOCATIONS**
**Target Directories for Shared Utilities:**
- `shared_utils/formatting.py` - Format and display utilities
- `shared_utils/conversion.py` - Data conversion utilities
- `shared_utils/helpers.py` - General helper functions
- `shared_utils/documentation.py` - Documentation generation utilities

## 🎯 EXTRACTION IMPLEMENTATION PLAN

### **GOLDCLAD PROTOCOL APPLICATION:**
Before creating any new utility files, execute mandatory similarity search:

1. **Search Existing Utilities**: 
   ```bash
   grep -r "shared_utils\|common_utils\|utilities" . --include="*.py"
   ```

2. **Check for Utility Directories**:
   - Look for existing `utils/`, `shared/`, `common/` directories
   - Examine existing modularization output directories

3. **Assess Enhancement vs Creation**:
   - If utilities exist: ENHANCE existing files
   - If no utilities found: CREATE with GOLDCLAD justification

### **STEELCLAD MODULARIZATION APPROACH:**
**For Large Files with Utilities:**
1. **Identify Utility Sections**: Functions that can be extracted
2. **Create Child Modules**: Separate utility modules from main logic
3. **Preserve Functionality**: Ensure all utility functions remain accessible
4. **Update Imports**: Maintain proper import relationships

## 🔄 UTILITY CONSOLIDATION TARGETS

### **IMMEDIATE TARGETS (Hour 3):**
1. **Extract from `enhanced_intelligence_linkage.py`**:
   - Helper functions → `shared_utils/helpers.py`
   - Format functions → `shared_utils/formatting.py`

2. **Consolidate Documentation Utilities**:
   - Generator utilities → `shared_utils/documentation.py`
   - Naming utilities → `shared_utils/naming.py`

### **SECONDARY TARGETS (Hour 4+):**
1. **Analysis Utilities**: Common analysis helper functions
2. **Integration Utilities**: Service and endpoint helpers
3. **Metrics Utilities**: Common metrics calculation functions

## 📊 SUCCESS METRICS

### **Utility Consolidation KPIs:**
- **Functions Extracted**: Target 15+ utility functions
- **Files Created**: 3-4 shared utility modules
- **Duplicates Eliminated**: Identify and consolidate duplicate utilities
- **Import Dependencies**: Simplify import chains through centralization

## 🚀 NEXT STEPS (Hour 3)
1. **Execute GOLDCLAD search** for existing utility infrastructure
2. **Apply STEELCLAD modularization** to target files
3. **Create/enhance shared utility modules** using identified patterns
4. **Document all consolidations** following established archive patterns

**UTILITY EXTRACTION PLAN**: ✅ COMPLETE  
**READY FOR EXECUTION**: ✅ Hour 3 implementation planned  
**LEVERAGE EXISTING**: ✅ ModularizationAutomator system identified