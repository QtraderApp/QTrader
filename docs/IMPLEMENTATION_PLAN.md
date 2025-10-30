# ManagerService Implementation Plan

**Last Updated:** October 30, 2025\
**Branch:** `feature/lego-architecture`\
**Objective:** Implement complete signal-to-order-to-fill event flow with Manager orchestrator and Risk tools library

______________________________________________________________________

## Architecture Overview

**Decision:** Manager owns all trading decisions (orchestrator), Risk is a library of pure stateless functions (calculators).

**Event Flow:**

```
Data → Strategy → Manager → Execution → Portfolio
       (PriceBar) (Signal)  (Order)    (Fill)    (State)
```

**Key Principles:**

- Manager is stateful orchestrator (subscribes to events, makes decisions, emits orders)
- Risk tools are pure functions (no state, no events, just calculations)
- No circular dependencies (Manager → Risk, one-way only)
- Complete audit trail (intent_id links signal → order → fill)

______________________________________________________________________

## Phase 1: Contract Updates & Idempotency

STATUS: completed ✅

**Objective:** Add idempotency and audit trail fields to event contracts

### Tasks

1. **Update OrderEvent Contract**

   - File: `src/qtrader/contracts/schemas/manager/order.v1.json`
   - Add `intent_id` (string, required): Links back to SignalEvent
   - Add `idempotency_key` (string, required): Replay protection key
   - Update example file: `order.v1.example.json`

1. **Update OrderEvent Pydantic Model**

   - File: `src/qtrader/events/events.py`
   - Add `intent_id: str` field to `OrderEvent` class
   - Add `idempotency_key: str` field to `OrderEvent` class
   - Update docstrings

1. **Update Contract Tests**

   - File: `tests/unit/contracts/test_order_schema.py` (create if missing)
   - Test schema validation with new required fields
   - Test idempotency_key format expectations

### Definition of Done

- [ ] OrderEvent schema validates with intent_id and idempotency_key
- [ ] OrderEvent Pydantic model includes new fields
- [ ] Contract tests pass (all existing + new tests)
- [ ] Example JSON validates against schema

**Time Estimate:** 1 hour

______________________________________________________________________

## Phase 2: Risk Tools Library (Pure Functions)

**Objective:** Extract calculation logic into pure stateless functions

### Tasks

1. **Extract Position Sizing Tools**

   - Source: `src/qtrader/services/manager/sizer.py`
   - Target: `src/qtrader/libraries/risk/tools/sizing.py`
   - Functions to extract:
     - `calculate_fixed_equity_size(equity, pct) → Decimal`
     - `calculate_fixed_quantity_size(quantity) → Decimal`
     - `calculate_volatility_target_size(...)` (future)
   - Make all functions pure (no class state, no side effects)

1. **Extract Limit Checking Tools**

   - Source: `src/qtrader/services/manager/limits.py`
   - Target: `src/qtrader/libraries/risk/tools/limits.py`
   - Functions to extract (already pure):
     - `check_concentration_limit(...) → tuple[bool, str]`
     - `check_leverage_limits(...) → tuple[bool, str]`
     - `check_all_limits(...) → tuple[bool, list[str]]`
   - Keep function signatures, just move files

1. **Create Risk Models Module**

   - Target: `src/qtrader/libraries/risk/models.py`
   - Extract from `services/manager/models.py`:
     - `RiskConfig` (configuration dataclass)
     - `ConcentrationLimit`, `LeverageLimit`, `SizingConfig`
   - Keep in manager: `PortfolioState`, `OrderBase`, `Signal` (manager-specific)

1. **Create Policy Loader**

   - Target: `src/qtrader/libraries/risk/loaders.py`
   - Function: `load_policy(name: str) → RiskConfig`
   - Search order:
     1. `src/qtrader/libraries/risk/builtin/{name}.yaml`
     1. `my_library/risk_policies/{name}.yaml`
   - Parse YAML into `RiskConfig` dataclass

1. **Update Risk Library Exports**

   - File: `src/qtrader/libraries/risk/__init__.py`
   - Export: sizing tools, limit tools, models, loaders
   - Remove: `BaseRiskPolicy` (old service-based pattern)

1. **Create User Policy Template**

   - File: `my_library/risk_policies/template.yaml`
   - Copy from `builtin/naive.yaml` with comments
   - Add instructions for customization

### Definition of Done

- [ ] All risk tools are pure functions (no state, no event bus)
- [ ] Tools can be imported from `qtrader.libraries.risk.tools`
- [ ] Policy loader reads builtin and custom policies
- [ ] User template provided with clear documentation
- [ ] 50+ unit tests for risk tools (90%+ coverage)

**Time Estimate:** 3 hours

______________________________________________________________________

## Phase 3: ManagerService Refactor

**Objective:** Transform RiskService into ManagerService orchestrator

### Tasks

1. **Rename Service Class**

   - File: `src/qtrader/services/manager/service.py`
   - Rename: `RiskService` → `ManagerService`
   - Update logger namespace: `"risk.service"` → `"manager.service"`
   - Update all docstrings and comments

1. **Update Event Subscriptions**

   - Subscribe to: `SignalEvent` (already exists)
   - Subscribe to: `ConsolidatedPortfolioEvent` (for portfolio state)
   - Remove: `RiskEvaluationTriggerEvent` (engine no longer needs to trigger)

1. **Refactor Signal Handler**

   - Method: `on_signal(event: SignalEvent)`
   - Generate `idempotency_key = f"{signal.strategy_id}-{signal.signal_id}-{signal.timestamp}"`
   - Call risk tools for sizing:
     ```python
     from qtrader.libraries.risk.tools.sizing import calculate_fixed_equity_size
     size = calculate_fixed_equity_size(equity, pct)
     ```
   - Call risk tools for limits:
     ```python
     from qtrader.libraries.risk.tools.limits import check_all_limits
     is_ok, reasons = check_all_limits(portfolio, symbol, size, config)
     ```
   - Create `OrderEvent` with:
     - `intent_id = signal.signal_id` (link back to signal)
     - `idempotency_key` (replay protection)
     - All order details from signal + sizing
   - Publish `OrderEvent` to event bus (not approval/rejection events)

1. **Update Configuration Loading**

   - Method: `from_config(config_dict, event_bus)`
   - Load policy from `portfolio.yaml`: `risk_policy.name`
   - Use `load_policy(name)` from risk loaders
   - Pass `RiskConfig` to ManagerService

1. **Remove Batch Evaluation**

   - Remove: `_signal_buffer` (process signals immediately)
   - Remove: `on_risk_evaluation_trigger()` method
   - Manager reacts to each signal instantly (event-driven, not batch)

1. **Update Manager Exports**

   - File: `src/qtrader/services/manager/__init__.py`
   - Rename: `RiskService` → `ManagerService`
   - Rename: `IRiskService` → `IManagerService`
   - Remove: Old approval/rejection models
   - Keep: `PortfolioState`, `OrderBase`, `Signal` (manager-specific)

1. **Update Manager Interface**

   - File: `src/qtrader/services/manager/interface.py`
   - Rename: `IRiskService` → `IManagerService`
   - Update method signatures to match new event flow

### Definition of Done

- [ ] Class renamed to ManagerService
- [ ] Emits OrderEvent (not approval/rejection)
- [ ] Generates idempotency_key and intent_id correctly
- [ ] Calls risk tools from libraries (not internal methods)
- [ ] Loads policy from portfolio.yaml
- [ ] Event-driven (no batch processing)
- [ ] 30+ unit tests passing (90%+ coverage)

**Time Estimate:** 4 hours

______________________________________________________________________

## Phase 4: BacktestEngine Integration

**Objective:** Wire ManagerService into the backtest engine event flow

### Tasks

1. **Initialize ManagerService**

   - File: `src/qtrader/engine/engine.py`
   - Uncomment ManagerService initialization (around line 332)
   - Load config from `BacktestConfig.risk_policy`
   - Create: `manager_service = ManagerService.from_config(config, event_bus)`
   - Store as instance variable: `self._manager = manager_service`

1. **Update Engine Lifecycle**

   - Add ManagerService to service initialization sequence
   - Ensure Manager subscribes to events before backtest starts
   - No need to explicitly trigger Manager (event-driven)

1. **Remove Temporary Workarounds**

   - File: `src/qtrader/services/strategy/context.py`
   - Remove: `emit_order()` method (strategies should only emit signals)
   - Keep: `emit_signal()` method only
   - Update docstrings to reflect signal-only emission

1. **Update Example Scripts**

   - File: `basic_run_example.py`
   - Ensure portfolio.yaml includes `risk_policy` section
   - Update comments to reflect new flow
   - File: `full_run_example.py`
   - Same updates

1. **Update Configuration Docs**

   - File: `config/README.md` (create if missing)
   - Document `risk_policy` section in portfolio.yaml
   - Document builtin policies and custom policy location

### Definition of Done

- [ ] ManagerService initialized in BacktestEngine
- [ ] Strategies only emit SignalEvent (no direct orders)
- [ ] Manager automatically processes signals and emits orders
- [ ] Example scripts run with new architecture
- [ ] Configuration documentation updated

**Time Estimate:** 2 hours

______________________________________________________________________

## Phase 5: Integration Testing

**Objective:** End-to-end tests verifying complete event flow

### Tasks

1. **Create Integration Test Structure**

   - Create directory: `tests/integration/`
   - Create: `tests/integration/__init__.py`
   - Create: `tests/integration/conftest.py` (shared fixtures)

1. **Test Phase 1: Signal to Order Flow**

   - File: `tests/integration/test_manager_signal_to_order.py`
   - Test: Data → Strategy → Manager → (mock Execution) → Portfolio
   - Verify:
     - SignalEvent published by strategy
     - OrderEvent published by manager with intent_id and idempotency_key
     - Order links back to signal via intent_id
     - Manager applies sizing correctly
     - Manager checks limits correctly
   - Use real services with minimal test data

1. **Test Policy Loading**

   - File: `tests/integration/test_manager_policy_loading.py`
   - Test: Load builtin naive policy
   - Test: Load custom policy from my_library/
   - Test: Policy configuration applied correctly
   - Verify: Manager uses correct sizing/limits from policy

1. **Test Idempotency**

   - File: `tests/integration/test_manager_idempotency.py`
   - Test: Same signal processed twice (same idempotency_key)
   - Verify: Manager only emits one OrderEvent
   - Test: Replay scenario with duplicate signals

1. **Test Multi-Strategy Coordination**

   - File: `tests/integration/test_manager_multi_strategy.py`
   - Test: Two strategies emit signals simultaneously
   - Verify: Manager tracks portfolio state correctly
   - Verify: Limits applied across all strategies
   - Verify: Capital allocation works correctly

### Definition of Done

- [ ] Integration test directory created
- [ ] 5+ integration tests covering end-to-end flows
- [ ] All integration tests pass
- [ ] Tests run in \<10 seconds total
- [ ] Clear test documentation

**Time Estimate:** 3 hours

______________________________________________________________________

## Phase 6: Documentation & Cleanup

**Objective:** Update documentation and remove deprecated code

### Tasks

1. **Update IMPLEMENTATION_STATUS.md**

   - Mark ManagerService as ✅ Complete
   - Mark Risk Library as ✅ Complete
   - Update "Current State" section
   - Update "How It Works" with actual implementation

1. **Create ManagerService Documentation**

   - File: `docs/lego_architecture/MANAGER_SERVICE.md`
   - Document architecture (orchestrator pattern)
   - Document event flow (signal → order)
   - Document policy loading
   - Document idempotency guarantees
   - Add code examples

1. **Create Risk Library Documentation**

   - File: `docs/lego_architecture/RISK_LIBRARY.md`
   - Document all risk tools (sizing, limits, etc.)
   - Document policy format (YAML schema)
   - Document custom policy creation
   - Add examples for each tool

1. **Update API Documentation**

   - File: `docs/API_DESIGN.md`
   - Add ManagerService API
   - Add Risk tools API
   - Update event flow diagrams

1. **Remove Deprecated Code**

   - Remove: Old approval/rejection event classes (if any)
   - Remove: Batch evaluation code
   - Remove: Any RiskService references in comments
   - Clean up: Unused imports

1. **Update README**

   - File: `README.md`
   - Update architecture diagram to show Manager
   - Update getting started guide
   - Add link to new documentation

### Definition of Done

- [ ] IMPLEMENTATION_STATUS.md updated
- [ ] ManagerService documented
- [ ] Risk library documented
- [ ] API docs updated
- [ ] Deprecated code removed
- [ ] README reflects current architecture

**Time Estimate:** 2 hours

______________________________________________________________________

## Summary Timeline

| Phase | Description                    | Time Estimate | Cumulative |
| ----- | ------------------------------ | ------------- | ---------- |
| 1     | Contract Updates & Idempotency | 1 hour        | 1 hour     |
| 2     | Risk Tools Library             | 3 hours       | 4 hours    |
| 3     | ManagerService Refactor        | 4 hours       | 8 hours    |
| 4     | BacktestEngine Integration     | 2 hours       | 10 hours   |
| 5     | Integration Testing            | 3 hours       | 13 hours   |
| 6     | Documentation & Cleanup        | 2 hours       | 15 hours   |

**Total Estimated Time:** ~15 hours (2 working days)

______________________________________________________________________

## Success Criteria

### Phase 1 (ManagerService) Complete When

- [ ] 30+ unit tests passing with 90%+ coverage
- [ ] 5+ integration tests passing
- [ ] Manager consumes SignalEvent and emits OrderEvent
- [ ] Orders include intent_id and idempotency_key
- [ ] Manager loads policies from portfolio.yaml
- [ ] Manager calls risk tools (not internal methods)
- [ ] BacktestEngine integration working
- [ ] Example scripts run successfully
- [ ] Documentation complete

### Overall Success

- [ ] All tests passing (unit + integration)
- [ ] No backward compatibility issues (clean refactor)
- [ ] Clear separation: Manager = orchestrator, Risk = tools
- [ ] Complete audit trail (signal → order → fill)
- [ ] Event-driven architecture (no batch processing)
- [ ] User can create custom policies in my_library/
- [ ] Ready for Phase 2: ExecutionService FSM

______________________________________________________________________

## Next Phase Preview

**Phase 2: ExecutionService FSM** (separate implementation plan)

- Delete old ExecutionEngine
- Implement FSM (NEW → ACK → PARTIAL → FILLED/CANCELED/REJECTED)
- Add idempotency checks (duplicate orders rejected)
- Emit FillEvent with source_order_id
- 40+ tests with 90%+ coverage

______________________________________________________________________

## Notes

- **No backward compatibility needed** - clean refactor allowed
- **No legacy code to maintain** - remove deprecated patterns
- **Full test coverage required** - 90%+ for all new code
- **Event-driven only** - no batch processing, no polling
- **Pure functions for risk tools** - easier to test and reason about
- **Complete audit trail** - every decision traceable via events
