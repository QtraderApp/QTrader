# Specification Document Consolidation Plan

**Date:** October 6, 2025 **Scope:** Streamline `docs/specs/phase01.md` from 2,242 lines to ~800 lines **Goal:** Keep authoritative reference, remove completed implementation details

______________________________________________________________________

## Current State Analysis

### Document Size Comparison

| Document                                 | Lines     | Status                       | Action             |
| ---------------------------------------- | --------- | ---------------------------- | ------------------ |
| `docs/specs/phase01.md`                  | 2,242     | Bloated with examples        | **Slim to ~800**   |
| `docs/implementation_plan_phase01.md`    | 2,841     | Outdated, too detailed       | **Delete**         |
| `docs/implementation_plan_phase01_v3.md` | 629       | Streamlined, current         | **Keep & rename**  |
| `STAGE_6B_IMPLEMENTATION_PLAN.md`        | 991       | Redundant (integrated in v3) | **Delete**         |
| **Total**                                | **6,703** |                              | **Target: ~1,400** |

### phase01.md Structure (20 sections)

| Section                     | Lines | Keep/Reduce   | Rationale                             |
| --------------------------- | ----- | ------------- | ------------------------------------- |
| 1. Purpose & Non-Goals      | 15    | ✅ Keep       | Core specification                    |
| 2. Architecture Overview    | 212   | 🔄 Reduce 50% | Too much schema detail, examples      |
| 3. Dataset Alignment        | 152   | 🔄 Reduce 40% | Excessive config examples             |
| 4. Orders & Execution       | 49    | ✅ Keep       | Core specification                    |
| 5. Shorting & Dividends     | 20    | ✅ Keep       | Core specification                    |
| 6. Calendar & Sessions      | 8     | ✅ Keep       | Core specification                    |
| 7. Event Loop               | 38    | 🔄 Reduce 30% | Keep algorithm, reduce examples       |
| 8. Configuration            | 40    | 🔄 Reduce 70% | Reference schema, cut examples        |
| 9. Costs & Commissions      | 12    | ✅ Keep       | Core specification                    |
| 10. Output Artifacts        | 37    | 🔄 Reduce 50% | Reference schema, cut examples        |
| 11. Error Handling          | 20    | ✅ Keep       | Core specification                    |
| 12. Testing Strategy        | 43    | ❌ Delete     | Implementation detail (covered in v3) |
| 13. Performance & Numerics  | 370   | 🔄 Reduce 80% | Keep principles, delete examples      |
| 14. Developer Notes         | 18    | ✅ Keep       | Core guidance                         |
| 15. Risk Management         | 438   | 🔄 Reduce 70% | Keep API, reduce examples             |
| 16. Glossary                | 11    | ✅ Keep       | Reference                             |
| 17. Phase-2 Backlog         | 14    | ✅ Keep       | Planning                              |
| 17. Appendix - JSON Schemas | 53    | 🔄 Reduce 80% | Reference external schemas            |
| 18. Worked Examples         | 43    | ❌ Delete     | Implementation detail                 |
| 19. Distribution & Usage    | 223   | ❌ Delete     | Implementation detail (not spec)      |
| 20. Interactive Debugging   | 466   | ❌ Delete     | Implementation detail (not spec)      |

**Total sections to delete:** 3 (732 lines) **Total sections to reduce:** 8 (~800 lines saved) **Target size:** ~710 lines (68% reduction)

______________________________________________________________________

## Problems Identified

### 1. **Specification vs Implementation Confusion**

**Problem:** `phase01.md` contains:

- ✅ GOOD: Architecture contracts, data models, API specifications
- ❌ BAD: CLI examples, debugging workflows, testing code snippets
- ❌ BAD: Full YAML configurations with comments
- ❌ BAD: Worked examples better suited for tutorials

**Example - Section 19 "Distribution & Usage" (223 lines):**

```markdown
## 19. Distribution & Usage (Installable Package + CLI)

### 19.1 Installation

pip install qtrader

### 19.2 CLI Reference

qtrader backtest --strategy strategies/my_strategy.py ...

### 19.3 Programmatic API

from qtrader import Backtest
bt = Backtest(...)
```

**This is NOT a specification - it's user documentation!**

### 2. **Excessive Code Examples**

**Problem:** Specification has 50+ code examples

- Section 13 (Performance): 370 lines, mostly example code
- Section 15 (Risk Management): 438 lines, mostly example code
- Section 20 (Debugging): 466 lines, all example code

**What specifications need:**

- ✅ Type signatures, contracts, schemas
- ✅ Invariants, constraints, rules
- ❌ NOT full implementation examples
- ❌ NOT tutorial-style walkthroughs

### 3. **Duplication with Implementation Plan**

**Problem:** Both documents cover testing, development workflow, CLI

- `phase01.md` Section 12: Testing Strategy (43 lines)
- `implementation_plan_v3.md`: Has complete test status
- **Duplication:** Maintenance burden, inconsistency risk

### 4. **Schema Bloat**

**Problem:** Inline schema examples make document hard to navigate

- Section 2.3: 97 lines of Python class definitions
- Section 8: 40 lines of YAML config examples
- Section 10: Full JSON schema listings

**Better approach:**

- Link to actual code files (single source of truth)
- Show simplified examples (5-10 lines max)
- Reference external schema docs

______________________________________________________________________

## Consolidation Strategy

### Phase 1: Delete Old Implementation Documents ✅

```bash
# Archive old documents
mkdir -p docs/archive
mv docs/implementation_plan_phase01.md docs/archive/implementation_plan_phase01_v2.3.md
mv STAGE_6B_IMPLEMENTATION_PLAN.md docs/archive/

# Activate v3 as main implementation plan
mv docs/implementation_plan_phase01_v3.md docs/implementation_plan_phase01.md
```

**Result:** 3,832 lines removed, single 629-line implementation plan remains

### Phase 2: Slim phase01.md Specification

**Principle:** Keep **WHAT** and **WHY**, remove **HOW**

#### 2A. Delete Implementation Sections (3 sections, -732 lines)

```
DELETE:
- Section 12: Testing Strategy (43 lines)
- Section 18: Worked Examples (43 lines)
- Section 19: Distribution & Usage (223 lines)
- Section 20: Interactive Debugging (466 lines)
```

**Rationale:**

- Testing strategy → covered in implementation plan
- Examples → better in tutorials/examples/ directory
- CLI usage → belongs in user documentation
- Debugging → developer guide, not specification

#### 2B. Reduce Example Bloat (8 sections, -800 lines)

**Section 2: Architecture (212 → 106 lines, -50%)**

- Keep: Port contracts, determinism rules, Bar/AdjustmentEvent types
- Remove: Full DataAdapter code (50 lines) → link to actual file
- Remove: Multi-dataset YAML examples (60 lines) → simplified 10-line example
- Remove: Schema mapping details (40 lines) → reference config docs

**Section 3: Dataset Alignment (152 → 91 lines, -40%)**

- Keep: Adjustment strategy, data modes, canonical Bar schema
- Remove: Full config examples (40 lines) → simplified 15-line example
- Remove: Validation policy details (20 lines) → link to validator code

**Section 7: Event Loop (38 → 27 lines, -30%)**

- Keep: Loop algorithm, phase descriptions
- Remove: Detailed pseudocode (11 lines) → high-level only

**Section 8: Configuration (40 → 12 lines, -70%)**

- Keep: Config schema reference
- Remove: Full YAML examples (28 lines) → link to examples/configs/

**Section 10: Output Artifacts (37 → 19 lines, -50%)**

- Keep: File list, schema references
- Remove: Full JSON examples (18 lines) → link to schema files

**Section 13: Performance & Numerics (370 → 74 lines, -80%)**

- Keep: Precision rules, rounding policies, performance targets
- Remove: Benchmark code (200 lines)
- Remove: Optimization examples (96 lines)

**Section 15: Risk Management (438 → 131 lines, -70%)**

- Keep: RiskSignal API, policy contracts, core rules
- Remove: Full strategy examples (180 lines)
- Remove: Config examples (127 lines)

**Section 17: Appendix - JSON Schemas (53 → 11 lines, -80%)**

- Keep: Schema file references
- Remove: Full inline schemas (42 lines) → link to schema files

**Total reduction: ~1,532 lines removed**

______________________________________________________________________

## Proposed New Structure (710 lines)

```markdown
# Equities Backtesting Engine — Specification (Phase 1, v2.0)

## Status Dashboard
- Version: 2.0 (streamlined)
- Last Updated: Oct 2025
- Implementation Status: Stage 6B Complete, Stage 7 Next

## 1. Purpose & Non-Goals (15 lines) ✅
- Core objectives
- Explicit non-goals

## 2. Architecture Overview (106 lines) 🔄
- 2.1 Ports & Adapters (contracts only)
- 2.2 Determinism (rules)
- 2.3 Bar Contract (canonical types, simplified examples)
- 2.4 Multi-Dataset Support (high-level, link to config docs)

## 3. Dataset Alignment (91 lines) 🔄
- 3.1 Canonical Bar (type definition)
- 3.2 Data Config (schema reference, simple example)
- 3.3 Integrity Checks (rules)
- 3.4 Validation Policies (high-level)

## 4. Orders & Execution (49 lines) ✅
- 4.1 Order Types
- 4.2 Time-In-Force
- 4.3 Fill Policy
- 4.4 Volume Participation
- 4.5 Order Lifecycle

## 5. Shorting & Dividends (20 lines) ✅
- Short borrow assumptions
- Dividend treatment (adjusted data)

## 6. Calendar & Sessions (8 lines) ✅
- Market calendar rules

## 7. Event Loop (27 lines) 🔄
- Loop algorithm (high-level)
- Phase descriptions

## 8. Configuration (12 lines) 🔄
- Schema reference
- Link to examples

## 9. Costs & Commissions (12 lines) ✅
- Commission model
- Fee calculation

## 10. Output Artifacts (19 lines) 🔄
- File list
- Schema references

## 11. Error Handling (20 lines) ✅
- Validation levels
- Error policies

## 12. Performance & Numerics (74 lines) 🔄
- Precision rules
- Performance targets
- Optimization constraints

## 13. Developer Notes (18 lines) ✅
- Code organization
- Naming conventions

## 14. Risk Management (131 lines) 🔄
- RiskSignal API
- Policy contracts
- Core rules

## 15. Glossary (11 lines) ✅
- Key terms

## 16. Phase-2 Backlog (14 lines) ✅
- Future enhancements

## 17. Appendix - Schemas (11 lines) 🔄
- Schema file references
```

**Total: ~710 lines (68% reduction)**

______________________________________________________________________

## Migration Steps

### Step 1: Backup Everything ✅

```bash
# Already planned in DOCUMENTATION_CONSOLIDATION.md
mkdir -p docs/archive
cp docs/specs/phase01.md docs/archive/phase01_v1.0.md
```

### Step 2: Delete Old Implementation Plans ✅

```bash
# Archive old implementation plans
mv docs/implementation_plan_phase01.md docs/archive/implementation_plan_phase01_v2.3.md
mv STAGE_6B_IMPLEMENTATION_PLAN.md docs/archive/STAGE_6B_IMPLEMENTATION_PLAN.md

# Activate v3
mv docs/implementation_plan_phase01_v3.md docs/implementation_plan_phase01.md
```

### Step 3: Slim phase01.md

```bash
# Create streamlined version
# (Will be done via file editing)
```

**Sections to delete:**

1. Section 12: Testing Strategy → DELETE
1. Section 18: Worked Examples → DELETE
1. Section 19: Distribution & Usage → DELETE
1. Section 20: Interactive Debugging → DELETE

**Sections to reduce:**

1. Section 2: Architecture → Remove full code examples
1. Section 3: Dataset → Simplify config examples
1. Section 7: Event Loop → High-level only
1. Section 8: Configuration → Reference only
1. Section 10: Output → Schema reference only
1. Section 13: Performance → Principles only
1. Section 15: Risk → API contracts only
1. Section 17: Appendix → References only

### Step 4: Update Cross-References

```bash
# Update any files referencing old docs
grep -r "implementation_plan_phase01.md" docs/ README.md
grep -r "STAGE_6B" docs/ README.md
```

### Step 5: Commit Changes

```bash
git add docs/ STAGE_6B_IMPLEMENTATION_PLAN.md
git commit -m "docs: Consolidate and streamline documentation

- Delete old implementation plans (3,832 lines → 629 lines)
- Slim phase01.md specification (2,242 → 710 lines)
- Archive old versions for reference
- Total reduction: 5,735 lines (79%)

Changes:
1. Deleted implementation sections from phase01.md:
   - Testing Strategy (covered in implementation plan)
   - Worked Examples (moved to examples/)
   - Distribution/CLI (user docs, not spec)
   - Interactive Debugging (dev guide, not spec)

2. Reduced example bloat in phase01.md:
   - Keep contracts, types, rules (WHAT/WHY)
   - Remove code examples (HOW - in actual code)
   - Link to external schemas and configs
   - Simplified examples (10 lines max)

3. Archived old implementation plans:
   - implementation_plan_phase01_v2.3.md (2,841 lines)
   - STAGE_6B_IMPLEMENTATION_PLAN.md (991 lines)

4. Activated streamlined v3:
   - implementation_plan_phase01.md (629 lines)
   - Integrated Stage 6B completion + extension

Result:
✅ Single source of truth for each concern
✅ Specification is authoritative reference (710 lines)
✅ Implementation plan tracks progress (629 lines)
✅ 79% documentation reduction
✅ No information loss (archived)
"
```

______________________________________________________________________

## Document Separation of Concerns

### phase01.md (Specification - 710 lines)

**Purpose:** Authoritative technical reference **Audience:** Architects, lead developers, external integrators **Content:**

- ✅ System contracts (APIs, types, protocols)
- ✅ Business rules (invariants, constraints)
- ✅ Architecture decisions (ports, adapters, determinism)
- ✅ Data models (Bar, Order, Position, etc.)
- ❌ NOT implementation examples
- ❌ NOT development workflows
- ❌ NOT user tutorials

**Update frequency:** Low (stable contracts)

### implementation_plan_phase01.md (Progress - 629 lines)

**Purpose:** Development roadmap and status tracker **Audience:** Development team, project managers **Content:**

- ✅ Stage completion status (dashboard)
- ✅ Test counts and coverage
- ✅ Current work (Stage 6B extension)
- ✅ Future stages (high-level plans)
- ✅ Architectural decisions made
- ❌ NOT detailed specifications
- ❌ NOT code examples for completed work

**Update frequency:** High (after each stage)

### README.md (Overview)

**Purpose:** Project introduction and quick start **Audience:** New contributors, users, stakeholders **Content:**

- ✅ What is QTrader?
- ✅ Quick start guide
- ✅ Installation instructions
- ✅ Link to detailed docs
- ❌ NOT detailed specifications
- ❌ NOT implementation details

### examples/ (Tutorials)

**Purpose:** Runnable code examples and tutorials **Audience:** Strategy developers, quant researchers **Content:**

- ✅ Example strategies (SMA crossover, etc.)
- ✅ Configuration examples
- ✅ Integration examples
- ❌ NOT specifications

______________________________________________________________________

## Success Metrics

### Quantitative

- [x] Reduce total docs from 6,703 → 1,339 lines (-80%)
- [x] phase01.md: 2,242 → 710 lines (-68%)
- [x] implementation_plan: 3,832 → 629 lines (-84%)
- [x] Archive old versions (no data loss)

### Qualitative

- [x] Clear separation of concerns (spec vs implementation vs tutorial)
- [x] Easier to find information (1/3 the pages to search)
- [x] Reduced duplication (single source of truth)
- [x] Lower maintenance burden (fewer examples to update)
- [x] Faster onboarding (less reading required)

______________________________________________________________________

## Recommendations

### Immediate Actions (Today)

1. ✅ **Delete old implementation plans** (Step 2)

   - Archive to `docs/archive/`
   - Activate v3 as main plan
   - **Time:** 5 minutes

1. ✅ **Slim phase01.md** (Step 3)

   - Delete Sections 12, 18, 19, 20 (-732 lines)
   - Reduce Sections 2, 3, 7, 8, 10, 13, 15, 17 (-800 lines)
   - **Time:** 1-2 hours

1. ✅ **Commit consolidated docs** (Step 5)

   - Single atomic commit
   - Clear commit message
   - **Time:** 5 minutes

**Total time:** 2 hours

### Future Improvements (Later)

1. **Create separate user documentation**

   - CLI reference guide (from Section 19)
   - Debugging guide (from Section 20)
   - Strategy development tutorial (from Section 18)
   - **Location:** `docs/guides/`

1. **Extract schemas to separate files**

   - JSON schemas → `docs/schemas/`
   - Config schemas → `examples/configs/schemas/`
   - Link from specification

1. **Create architecture diagrams**

   - Port/adapter architecture
   - Event loop flow
   - Order lifecycle state machine
   - **Location:** `docs/diagrams/`

______________________________________________________________________

## Approval Checklist

Before proceeding:

- [ ] Review archive strategy (keep old versions?)
- [ ] Approve deletion of Sections 12, 18, 19, 20 from phase01.md
- [ ] Approve reduction strategy for Sections 2, 3, 7, 8, 10, 13, 15, 17
- [ ] Review new document structure (710 + 629 = 1,339 lines total)
- [ ] Approve commit message

______________________________________________________________________

**Next Steps:** Awaiting approval to proceed with consolidation.
