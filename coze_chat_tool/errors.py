class CozeAPIError(Exception):
    """Base exception for all Coze API errors."""
    def __init__(self, message: str, status_code: int | None = None, code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class CozeAuthError(CozeAPIError):
    """401 - Authentication failed."""


class CozePermissionError(CozeAPIError):
    """403 - Permission denied."""


class CozeNotFoundError(CozeAPIError):
    """404 - Resource not found."""


class CozeRateLimitError(CozeAPIError):
    """429 - Rate limit exceeded."""


class CozeNetworkError(Exception):
    """Network timeout, DNS failure, or connection error."""
