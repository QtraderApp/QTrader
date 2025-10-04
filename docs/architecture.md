# QTrader Architecture Overview

**High-level system architecture and component diagrams**

For detailed architecture diagrams with implementation status visualization, see:

**[📊 Architecture Diagrams](diagrams/architecture.md)**

This includes:

- **System Architecture Overview** - Complete component hierarchy with color-coded implementation status
- **Event Loop Architecture** - Sequence diagram showing strategy lifecycle and warmup phases
- **Data Adapter Architecture** - Vendor normalization to canonical Bar model
- **Order Execution Flow** - End-to-end order processing with fill policies
- **Indicator Framework Architecture** - Planned indicator system (Stage 6A)
- **Implementation Progress** - Detailed status tables for all 8 stages
- **Component Dependencies** - Stage dependency graph

## Quick Reference

### Implementation Status (Color-Coded)

- 🟢 **Green**: Implemented & tested (Stages 1-4 complete)
- 🟡 **Yellow**: Specified but not yet implemented (Stages 5-8 planned)
- 🔵 **Blue**: External dependencies or data sources

### Current Progress

**Completed:** Stages 1-5A (177 tests passing)

- ✅ Data models & adapters (Bar, Order, Portfolio, Ledger)
- ✅ Execution engine (Market, MOC, Limit, Stop orders)
- ✅ Commission models
- ✅ Strategy base class & Context API
- ✅ Volume participation with partial fills

**In Progress:** Stage 5B (Risk Management System)

- 🔄 Signal model (trading intent before sizing)
- 🔄 RiskManager (portfolio-scoped evaluation)
- 🔄 Position sizing methods (4 basic methods)
- 🔄 Concentration & leverage limits
- 🔄 Strategy integration (signal-based workflow)

**Planned:** Stages 6A-8

- 🔄 Indicator framework with warmup system (Stage 6A)
- 🔄 Shorting & borrow fees (Stage 6B)
- 🔄 Public API & CLI (Stage 7)
- 🔄 Golden baseline tests (Stage 8)

## Related Documentation

- **Technical Specification:** [specs/phase01.md](specs/phase01.md)
- **Implementation Plan:** [implementation_plan_phase01.md](implementation_plan_phase01.md)
- **Logging Guide:** [logging.md](logging.md)
