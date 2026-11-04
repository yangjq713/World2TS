"""Exception hierarchy for World2TS-core."""


class World2TSError(Exception):
    """Base exception. All CLI errors inherit from this."""


class SourceError(World2TSError):
    """External API or data source failures."""


class StoreError(World2TSError):
    """Artifact store I/O failures."""
