class ExecutionError(RuntimeError):
    """Execution failed; the model session must be closed."""


class CancelledError(ExecutionError):
    """Execution was cancelled; the model session must be closed."""
