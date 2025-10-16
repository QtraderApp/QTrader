# Phase 2 Kickoff Summary

**Date:** October 16, 2025 **Branch:** `feature/lego-phase2-portfolio-service` **Status:** 📝 Planning Complete, Ready to Implement

## What Was Decided

We've created a comprehensive Phase 2 specification by analyzing and combining two approaches:

- **v1 (Original)**: Simple, follows Lego pattern, but lacked accounting rigor
- **v2 (External)**: Production-grade double-entry accounting, but complex
- **v3 (Final)**: Best of both - Lego pattern + essential accounting rigor

## Key Design Decisions

### 1. **Feed Philosophy** (Critical!)

- ✅ **Treat ALL feeds as unadjusted** - No "feed awareness"
- ✅ Portfolio sees prices "as of that point in time"
- ✅ User responsible if using adjusted feed without corp actions
- ✅ Eliminates dual-mode complexity

### 2. **Lot Accounting**

- ✅ **FIFO for long positions** (First In, First Out)
- ✅ **LIFO for short positions** (Last In, First Out)
- ✅ Accurate realized P&L tracking
- ✅ From the beginning - build it right

### 3. **Scope**

Must support:

- Long/short positions with full lifecycle
- Entry/exit fees (per-share and percentage)
- Dividend income (longs) and expense (shorts)
- Borrow fees on shorts (daily accrual)
- Margin interest on negative cash (daily accrual)
- Stock splits and reverse splits
- Complete ledger and audit trail

Deferred:

- Stock dividends (non-cash)
- Partial fills
- Mergers, spinoffs, rights issues
- Multi-currency

### 4. **Accounting Principles**

- ✅ Commissions tracked **separately** (NOT in cost basis)
- ✅ Daily mark-to-market
- ✅ Daily fee/interest accrual
- ✅ Keep position history (flat positions retained)
- ✅ Deterministic replay from ledger

### 5. **Edge Cases Handled**

- Position transitions: long → flat → short (same symbol)
- Splits on short positions
- Missing prices (use last known)
- Zero equity (leverage undefined)

## Implementation Timeline

### Week 1: Core Service + Ledger

- Basic fill processing (open long/short)
- Ledger implementation
- Cash management
- Basic queries

**Deliverable:** Can open positions and track cash

### Week 2: Lot Accounting + P&L

- FIFO/LIFO lot matching
- Closing positions with realized P&L
- Position transitions
- P&L calculations

**Deliverable:** Full position lifecycle with accurate P&L

### Week 3: Corporate Actions + Fees

- Stock splits (regular and reverse)
- Dividends (long receive, short pay)
- Borrow fees (daily accrual)
- Margin interest (daily accrual)
- Mark-to-market orchestration

**Deliverable:** Complete corporate actions and fee handling

### Week 4: State Management + Polish

- Portfolio state snapshots
- Edge case handling
- Mock implementation
- Comprehensive testing (>90% coverage)
- Documentation

**Deliverable:** Production-ready PortfolioService

## Success Criteria

### Functional

- [ ] Can process all transaction types
- [ ] FIFO/LIFO lot matching works correctly
- [ ] Realized and unrealized P&L accurate
- [ ] Corporate actions process correctly
- [ ] Fees and interest accrue properly
- [ ] Complete ledger maintained

### Technical

- [ ] Implements `IPortfolioService` protocol
- [ ] Zero dependencies on other services
- [ ] MyPy clean
- [ ] Test coverage ≥ 90%
- [ ] All tests pass
- [ ] Mock implementation available

### Invariants

- [ ] Equity = Cash + Position Values
- [ ] Splits preserve total value
- [ ] Ledger chronologically ordered
- [ ] All Decimal arithmetic

## Why This Matters

PortfolioService is the **most critical service** in QTrader:

1. **Source of truth** for all portfolio state
1. **Determines P&L** - must be accurate for strategy evaluation
1. **Foundation** for risk management (Phase 4)
1. **Enables** realistic backtesting with fees, interest, corporate actions

Getting this right is essential for the entire framework.

## Next Steps

1. ✅ Specification complete (this document)
1. 🚀 Begin Week 1 implementation
1. 📝 Create detailed task breakdown
1. 💻 Start coding!

## Documents

- **Main Spec:** `phase2_portfolio_service_v3.md` (1100+ lines)
- **Original v1:** `phase2_portfolio_service.md` (reference)
- **External v2:** `phase2_portfolio_service V2.md` (reference)

______________________________________________________________________

**Status:** Ready to implement! 🎯 **Confidence:** High - requirements are crystal clear **Complexity:** Medium-High, but well-scoped **Timeline:** 4 weeks with clear milestones

Let's build the most important service in QTrader! 💪
