"""Errors safe to return to API clients."""


class ResearchError(Exception):
    """Base domain error."""

    status_code = 400


class NotFound(ResearchError):
    """Requested paper does not exist."""

    status_code = 404


class Conflict(ResearchError):
    """Operation conflicts with ingestion already in progress."""

    status_code = 409


class ProviderError(ResearchError):
    """A model provider could not complete a request."""

    status_code = 503
