"""Runtime exceptions for the standalone NR worker."""
class NRError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status

class NREnvironmentError(NRError):
    """Runtime/device/worker infrastructure failure, not a bad input file."""
