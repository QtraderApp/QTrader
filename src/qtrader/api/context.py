"""Context object passed to strategy methods."""


class Context:
    """
    Context object providing strategy interface to engine.

    Stub for Stage 1. Full implementation in later stages.
    """

    def __init__(self):
        # TODO: Implement in later stages
        pass

    def buy_market(self, qty: int) -> str:
        """Submit market buy order."""
        raise NotImplementedError("Stage 1: Context stub only")

    def sell_market(self, qty: int) -> str:
        """Submit market sell order."""
        raise NotImplementedError("Stage 1: Context stub only")
