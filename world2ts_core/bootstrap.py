"""Central bootstrap — import all adapters so their @register decorators fire.

Any module that needs to look up sources via the registry should import this
first (or call ``ensure_sources_registered()``) to guarantee the registry is
fully populated.

"""

from __future__ import annotations

_bootstrapped = False


def ensure_sources_registered() -> None:
    """Idempotently import every adapter module so decorators run."""
    global _bootstrapped
    if _bootstrapped:
        return

    # Web-search sources used by Greenfield offline discovery.
    from world2ts_core.sources import tavily as _tavily  # noqa: F401
    from world2ts_core.sources import anysearch as _anysearch  # noqa: F401

    # Series sources
    from world2ts_core.series import yahoo as _yahoo  # noqa: F401
    from world2ts_core.series import fred as _fred  # noqa: F401
    from world2ts_core.series import binance as _binance  # noqa: F401
    from world2ts_core.series import noaa as _noaa  # noqa: F401
    from world2ts_core.series import noaa_isd as _noaa_isd  # noqa: F401
    from world2ts_core.series import owid as _owid  # noqa: F401
    from world2ts_core.series import eia as _eia  # noqa: F401
    from world2ts_core.series import polymarket as _polymarket  # noqa: F401

    _bootstrapped = True
