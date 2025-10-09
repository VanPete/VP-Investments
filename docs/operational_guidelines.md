# VP Investments - Operational Guidelines
*Development Framework for Consolidated Backend Structure*

**Last Updated:** 2025-10-08  
**Status:** Active Development Framework

---

## 👥 Role Definition

### User Role: Project Orchestrator
- Provides ideas, features, and strategic direction
- Makes architectural decisions and approves changes
- Reviews results and guides project priorities
- Defines business logic and requirements

### AI Agent Role: Implementation Engineer
- Implements features based on user requirements
- Makes tactical coding decisions within guidelines
- Tests and validates changes thoroughly
- Uses `tables.py` to understand database schema before making changes
- Updates documentation after implementations

**Key Principle**: User provides the "what" and "why" - AI determines the "how"

---

## 🎯 Project Structure Overview

### Backend Architecture (Core Development Area)
```
backend/                    # Main consolidated codebase - MODIFY EXISTING FILES ONLY
├── api/                   # API endpoints and web services
│   ├── api.py            # Main API implementation
│   └── __init__.py
├── core/                 # Core business logic and configuration
│   ├── backtest.py       # Backtesting engine
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Configuration management
│   ├── core.py           # Core enums and exceptions
│   ├── intelligence.py   # AI intelligence processing
│   ├── signals.py        # Signal processing logic
│   └── __init__.py
├── integrations/         # External service integrations
│   ├── ai.py             # AI service integration (OpenAI)
│   ├── ai_strategy_generator.py  # AI strategy generation
│   ├── backtest.py       # Integration-specific backtesting
│   ├── news.py           # News data integration
│   ├── production.py     # Production environment setup
│   ├── reddit.py         # Reddit scraping integration (PRAW)
│   ├── scheduler.py      # Task scheduling
│   ├── signal_processing.py  # Signal enhancement processing
│   ├── yfinance.py       # Yahoo Finance integration
│   └── __init__.py
├── storage/              # Supabase database interactions
│   ├── database.py       # Database operations and connections
│   └── __init__.py
├── utils/                # Utility functions and helpers
│   ├── logger.py         # Logging utilities
│   ├── observability.py  # Monitoring and observability
│   └── __init__.py
├── pipeline.py           # Main UnifiedPipeline - PRIMARY ENTRY POINT
├── py.typed              # Type checking marker
└── __init__.py
```

### Root Directory (Testing & Temporary Files)
```
root/                      # TEMPORARY FILES ONLY - Will be deleted when project complete
├── *.py                  # Temporary test scripts (allowed)
├── *.bat                 # Utility batch files (allowed)
├── requirements.txt      # Dependencies
├── pyproject.toml        # Project configuration
├── migrations/           # Database schema migrations
└── [other temp files]    # Testing utilities, one-off scripts
```

### Documentation (docs/)
```
docs/                      # Project documentation - READ BEFORE DEVELOPMENT
├── recommendations.md    # ⭐ SINGLE SOURCE OF TRUTH - ALL recommendations go here
├── operational_guidelines.md  # This file - development framework
└── [additional docs]     # Technical documentation and guides
```

**CRITICAL DOCUMENTATION RULE:**
- ⭐ **ALL recommendations, priorities, and status updates MUST go in `docs/recommendations.md`**
- **NEVER create separate recommendation files** (e.g., OPTIMIZATION_RECOMMENDATIONS.md, PRIORITY_GUIDE.md)
- **NEVER put recommendations in root directory**
- `recommendations.md` is a **living document** - update it as project progresses
- This is the **ONLY place** for recommendations, priorities, and implementation guidance

---

## 🏗️ Architectural Principles

### 1. Single Source of Truth
**Principle:** One primary data source, all other representations derive from it

**Implementation:**
- `signals` table is the primary source for all signal data
- `signals_norm` is a materialized view derived from `signals`
- Pipeline writes **only to `signals`**, views refresh automatically
- No duplicate data storage

**Example:**
```python
# ❌ OLD WAY: Duplicate writes
await db.insert_signals(signals)
await db.insert_signals_norm(normalized_signals)  # Duplicate data

# ✅ NEW WAY: Single write, derived view
await db.insert_signals(signals)
await db.refresh_materialized_view('signals_norm')  # Derives from signals
```

### 2. Commentary Consolidation
**Principle:** Unified commentary field for frontend simplification

**Implementation:**
- Single `commentary` field in `signals` table
- Structured format: **Signal Analysis** → **Market Insights** → **Key Metrics**
- `commentary_metadata` JSONB field tracks generation details
- Backward compatible (old fields kept during migration)

**Example:**
```python
# Backend generates unified commentary
commentary = self._generate_unified_commentary(
    signal=signal_data,
    score_explanation=basic_explanation,
    ai_commentary=ai_analysis  # Optional, only for top signals
)

# Frontend uses single field
signal.commentary  # Complete, formatted commentary
```

### 3. Top-N AI Commentary Pattern
**Principle:** Full AI commentary for highest-value signals only

**Implementation:**
- Top 10 signals (sorted by `weighted_score`) get full AI commentary
- Remaining signals get basic commentary (no AI call)
- 73% reduction in API calls, 52.7% performance improvement

**Example:**
```python
# Sort signals by weighted_score
top_signals = sorted(signals, key=lambda s: s.weighted_score, reverse=True)[:10]

# Full AI commentary for top 10
for signal in top_signals:
    signal.ai_commentary = await ai_service.generate_commentary(signal)

# Basic commentary for remaining signals
for signal in remaining_signals:
    signal.commentary = generate_basic_commentary(signal)
```

### 4. Materialized View Pattern
**Principle:** Derived views for performance without data duplication

**Implementation:**
- Use materialized views for expensive queries
- Refresh after data changes (automated in pipeline)
- Indexes on materialized views for query performance

**Example:**
```sql
-- Create materialized view
CREATE MATERIALIZED VIEW signals_norm AS
SELECT id, ticker, signal_type, weighted_score, ...
FROM signals;

-- Create indexes for performance
CREATE INDEX idx_signals_norm_ticker ON signals_norm(ticker);

-- Refresh after pipeline runs
REFRESH MATERIALIZED VIEW signals_norm;
```

---

## 🚀 Development Rules & Constraints

### ✅ ALLOWED Operations

#### 1. Backend File Modifications
- **EXTEND existing files** in `backend/` when adding features
- **MODIFY existing functions/classes** to fix issues or enhance functionality
- **ADD new functions/methods** to existing backend files
- **UPDATE imports** within backend files to maintain consistency
- **FOLLOW architectural principles** (single source of truth, commentary consolidation)

#### 2. Root Directory Usage
- **CREATE temporary test scripts** for feature validation
- **WRITE utility scripts** for one-off operations (e.g., `refresh_signals_norm.py`)
- **ADD batch files** for automation during development
- **CREATE test data files** for validation purposes
- **ADD migration files** in `migrations/` directory

#### 3. Testing & Validation
- **ALWAYS test using existing pipeline**: `from backend.pipeline import UnifiedPipeline`
- **RUN integration tests** after modifications
- **VALIDATE database connections** through `backend.storage.database`
- **TEST API endpoints** via `backend.api.api`
- **USE utility scripts** for validation (e.g., `tables.py --detailed`)

#### 4. Database Operations
- **CREATE migrations** for schema changes in `migrations/` directory
- **USE CASCADE** when dropping dependent objects
- **ADD indexes** for performance-critical queries
- **REFRESH materialized views** after data changes
- **VALIDATE data quality** with quality check scripts

### ❌ RESTRICTED Operations

#### 1. Backend Structure Changes
- **NO new files** in `backend/` without explicit approval
- **NO directory restructuring** within backend
- **NO deletion** of existing backend files
- **NO breaking changes** to core interfaces
- **NO duplicate data storage** (follow single source of truth)

#### 2. Data Architecture Violations
- **NO writing to derived views/tables** (only to primary sources)
- **NO data duplication** across tables
- **NO bypassing materialized view refresh** after writes
- **NO direct manipulation of signals_norm** (it's a view)

#### 3. Dependency Management
- **ASK before adding** new Python packages
- **CONFIRM before modifying** existing integrations
- **VALIDATE compatibility** with Supabase storage layer
- **CONSIDER async alternatives** (e.g., Async PRAW instead of PRAW)

---

## 🛠 Development Workflow

### Feature Development Process
1. **Read Documentation**: Check `docs/recommendations.md` for current status and priorities
2. **Identify Target File**: Determine which existing backend file needs modification
3. **Follow Architectural Principles**: Apply single source of truth, commentary consolidation, etc.
4. **Extend Functionality**: Add new methods/functions to existing files
5. **Update Imports**: Ensure all imports use `backend.*` structure
6. **Create Migration**: If database changes needed, create migration in `migrations/`
7. **Test Integration**: Use `UnifiedPipeline` for end-to-end testing
8. **Validate Storage**: Confirm Supabase interactions work correctly
9. **⭐ UPDATE RECOMMENDATIONS.MD**: **ALWAYS** update `docs/recommendations.md` with changes, decisions, and status
   - Add new priorities to the Pending Priorities section
   - Update completed tasks in the appropriate sections
   - Document architectural decisions
   - Add questions/blockers that need resolution
   - **NEVER create separate recommendation files**

### Testing Protocol
```python
# Standard Testing Pattern (create in root as temporary file)
from backend.pipeline import UnifiedPipeline

# Initialize pipeline
pipeline = UnifiedPipeline()

# Test your modifications
await pipeline.run()  # Full pipeline test

# Or test specific step
signals = await pipeline._generate_signals()

# Validate results
print("✅ Backend modifications successful!")
```

### Database Migration Protocol
```sql
-- migrations/XXX_description.sql

-- Part 1: Schema Changes
ALTER TABLE table_name ADD COLUMN new_column TYPE;

-- Part 2: Data Backfill (if needed)
UPDATE table_name SET new_column = ...;

-- Part 3: Drop Dependencies with CASCADE
DROP VIEW IF EXISTS dependent_view CASCADE;

-- Part 4: Create Derived Objects
CREATE MATERIALIZED VIEW new_view AS ...;

-- Part 5: Performance Indexes
CREATE INDEX idx_name ON table_name(column_name);

-- Part 6: Validation
SELECT COUNT(*) FROM table_name WHERE new_column IS NULL;
```

### Code Modification Guidelines
- **File Selection**: Always extend existing backend files rather than creating new ones
- **Function Placement**: Add new functionality to the most appropriate existing module
- **Import Updates**: Maintain `backend.*` import structure throughout
- **Error Handling**: Follow existing error handling patterns in backend files
- **Commentary Generation**: Use `_generate_unified_commentary()` for all commentary
- **Database Writes**: Write to primary tables only, let views derive data
- **View Refresh**: Always refresh materialized views after primary table updates

---

## 📁 Primary Work Areas

### Core Business Logic
- `backend/core/` - Main business logic, configuration, CLI
- `backend/pipeline.py` - **PRIMARY ENTRY POINT** and workflow orchestration
  - Step 1: Reddit data scraping
  - Step 2: Signal preprocessing
  - Step 3: Signal validation
  - Step 4: Signal scoring and generation
  - Step 4.6: **Unified commentary generation** (top 10 only)
  - Step 5: AI strategy generation
  - Step 6: Database persistence and view refresh

### Data & Integrations  
- `backend/integrations/` - External service connections
  - `reddit.py` - Reddit scraping (PRAW) - **Consider migrating to Async PRAW**
  - `yfinance.py` - Yahoo Finance integration
  - `ai.py` - OpenAI integration (commentary, strategies)
  - `news.py` - News data integration
  - `backtest.py` - Backtesting integration
- `backend/storage/database.py` - **Supabase database operations**
  - All database writes go here
  - Materialized view refresh methods
  - Query optimization

### API & Services
- `backend/api/api.py` - Web endpoints and external interfaces
- `backend/utils/` - Logging, observability, and utility functions

---

## 🔄 Integration Points

### Database Operations (Supabase)
```python
from backend.storage.database import SupabaseStorage

# Initialize storage
storage = SupabaseStorage()

# Write to primary table (signals)
await storage.insert_signals(signals)

# Refresh derived view (signals_norm)
await storage.refresh_materialized_view('signals_norm')

# ❌ Don't write to derived views
# await storage.insert_signals_norm(...)  # WRONG!
```

### Pipeline Integration
```python
from backend.pipeline import UnifiedPipeline

# Initialize and run
pipeline = UnifiedPipeline()
await pipeline.run()

# Access specific steps
signals = await pipeline._generate_signals()
commentary = pipeline._generate_unified_commentary(signal)
```

### External Services
```python
from backend.integrations.reddit import RedditIntegration
from backend.integrations.yfinance import YFinanceIntegration
from backend.integrations.ai import AIIntegration

# Service integrations through dedicated modules
reddit = RedditIntegration()
yfinance = YFinanceIntegration()
ai = AIIntegration()
```

### Commentary Generation
```python
# In backend/pipeline.py
def _generate_unified_commentary(
    self,
    signal: Dict[str, Any],
    score_explanation: str,
    ai_commentary: Optional[str] = None
) -> str:
    """Generate unified commentary following architectural pattern"""
    
    # Structured format
    sections = []
    
    # Signal Analysis (always present)
    sections.append(f"📊 **Signal Analysis**\n{score_explanation}")
    
    # Market Insights (if AI commentary available)
    if ai_commentary:
        sections.append(f"\n\n🔍 **Market Insights**\n{ai_commentary}")
    
    # Key Metrics (always present)
    sections.append(f"\n\n📈 **Key Metrics**\n- Score: {signal['weighted_score']}")
    
    return "\n".join(sections)
```

---

## ⚡ Quick Reference Commands

### Test Pipeline Integration
```bash
cd "c:\Users\willi\OneDrive\Desktop\Python Projects\VP Investments"
python -c "from backend.pipeline import UnifiedPipeline; pipeline = UnifiedPipeline(); print('✅ Pipeline operational')"
```

### Validate Backend Structure
```bash
python -c "import backend; print('✅ Backend imports successful')"
```

### Run Full Pipeline Test
```bash
python -m backend.pipeline
```

### Validate Data Quality
```bash
python tables.py --detailed
```

### Refresh Materialized Views
```bash
python refresh_signals_norm.py
```

### Clear Test Data (Safe)
```bash
python safe_clear_data.py
```

---

## 📊 Current Implementation Status

### Completed Architectural Patterns
- ✅ **Single Source of Truth**: signals table primary, signals_norm derived
- ✅ **Commentary Consolidation**: Unified commentary field with metadata
- ✅ **Top-N AI Pattern**: Top 10 signals get full AI commentary
- ✅ **Materialized View Pattern**: signals_norm as materialized view

### Active Development Areas
- 🔄 **Backtest Integration**: Populating performance tables (Priority #4)
- 🔄 **Performance Optimization**: Caching, parallel processing, async PRAW
- 🔄 **Data Quality**: Enhanced validation and monitoring

### Pending Architectural Decisions
- ⏳ **Frontend Migration**: Transition to use `commentary` field
- ⏳ **View Freshness Monitoring**: Automated alerts for stale views
- ⏳ **Backtest Scheduling**: Automated performance tracking
- ⏳ **Market Conditions**: Contextual data population

---

## 🚨 Important Reminders

### 📝 Documentation (CRITICAL)
- **⭐ SINGLE RECOMMENDATIONS FILE**: ALL recommendations MUST go in `docs/recommendations.md`
- **NO SEPARATE FILES**: Never create OPTIMIZATION_RECOMMENDATIONS.md, PRIORITY_GUIDE.md, etc.
- **UPDATE AFTER EVERY CHANGE**: Update recommendations.md with status, decisions, blockers
- **LIVING DOCUMENT**: recommendations.md evolves with the project

### Database Operations
- **PRIMARY TABLE ONLY**: Write to `signals`, not `signals_norm`
- **REFRESH VIEWS**: Always refresh materialized views after writes
- **CASCADE DROPS**: Use CASCADE when dropping objects with dependencies
- **VALIDATE CHANGES**: Run validation queries after migrations

### Code Architecture
- **SINGLE SOURCE**: One primary data source, derived representations only
- **UNIFIED COMMENTARY**: Use `_generate_unified_commentary()` method
- **TOP-N PATTERN**: Full AI for top signals, basic for others
- **NO DUPLICATION**: Never duplicate data across tables

### Development Process
- **READ DOCS FIRST**: Check `docs/recommendations.md` before starting
- **TEST THOROUGHLY**: Use `UnifiedPipeline` for integration testing
- **⭐ UPDATE RECOMMENDATIONS.MD**: **MANDATORY** after every change
- **ASK BEFORE**: Consult before major architectural changes

### Performance Considerations
- **MATERIALIZED VIEWS**: Use for expensive queries
- **INDEXES**: Add indexes for frequently queried columns
- **API LIMITS**: Top-N pattern for expensive API calls
- **ASYNC PREFERRED**: Use async libraries when available

---

## 🎨 Frontend Development (Future)
- **Separate structure** to be created when backend is complete
- **Independent from backend** - will import backend as dependency
- **Uses `commentary` field** for simplified rendering
- **Queries `signals_norm` view** for performance
- **New file creation allowed** in frontend directory structure
- **Different development rules** will apply

---

## 📚 Additional Resources

### Documentation
- **Recommendations**: `docs/recommendations.md` - Living document with priorities
- **This File**: `docs/operational_guidelines.md` - Development framework
- **Migrations**: `migrations/` - Database schema change history

### Utility Scripts

#### tables.py - Database Schema Inspector (PRIMARY TOOL)
**Purpose:** Comprehensive Supabase schema inspection and analysis

**Usage:**
```bash
# Interactive mode (recommended for exploration)
python tables.py

# List all tables with row counts
python tables.py --list

# Show table schema
python tables.py --schema signals

# Analyze NULL coverage and data quality
python tables.py --nulls signals

# Get optimization recommendations
python tables.py --recommend

# Generate full report
python tables.py --report

# Export report to file
python tables.py --export schema_report.md
```

**When to Use:**
- ✅ Before making any database changes (understand current state)
- ✅ When planning migrations (identify issues)
- ✅ During debugging (check data quality)
- ✅ For documentation (generate schema reports)
- ❌ Do NOT create new check scripts - use tables.py instead

**Importable Functions** (for use in other scripts):
```python
from tables import (
    check_table_exists,      # Check if table exists
    get_row_count,           # Get table row count
    get_column_names,        # List all columns
    check_column_exists,     # Check if column exists
    get_table_schema,        # Get full schema details
    analyze_column_nulls     # Get NULL coverage stats
)

# Example usage
if check_table_exists('signals'):
    row_count = get_row_count('signals')
    columns = get_column_names('signals')
    print(f"signals table has {row_count} rows and {len(columns)} columns")
```

**Features:**
- Lists all tables with row counts and status
- Shows detailed schema (columns, types, constraints)
- Analyzes NULL coverage and data quality
- Identifies redundant/useless columns
- Detects constant values and low variance
- Recommends schema optimizations
- Exports reports in Markdown format

**Recommendations Engine:**
- ❌ DROP EMPTY TABLE - 0 rows, no data
- ❌ DROP NULL COLUMN - 100% NULL, no useful data
- ⚠️ CONSTANT COLUMN - 100% same value, verify calculation
- ⚠️ LOW VARIANCE - <5% unique values, check data quality
- ⚠️ HIGH NULL RATE - >80% NULL, improve data collection
- 🔄 REDUNDANT COLUMNS - combine or remove duplicates
- 🔄 CALCULATED COLUMN - can derive from other columns
- ❌ REDUNDANT TABLE - duplicates another table's data

#### Other Utility Scripts
- `refresh_signals_norm.py` - Manual materialized view refresh
- `safe_clear_data.py` - Safe data deletion for testing

### Key Files
- `backend/pipeline.py` - Main entry point (line 1861: `_generate_unified_commentary`)
- `backend/storage/database.py` - Database operations
- `migrations/001_uuid_and_commentary_fixes.sql` - Recent schema changes

---

*Follow these guidelines to maintain the consolidated backend structure while enabling flexible development within the established architectural framework. Always prioritize single source of truth, unified commentary, and materialized view patterns.*

**Last Updated:** 2025-10-04 - Added architectural principles and current implementation status
