"""Official holiday dates used by Greenfield offline discovery."""

from __future__ import annotations

from datetime import date

import holidays as python_holidays


def country_holiday_dates(
    country_code: str,
    subdivision_code: str,
    start: date,
    end: date,
) -> list[tuple[date, str]]:
    """Return official holidays for one explicit country/subdivision."""

    calendar = python_holidays.country_holidays(
        country_code,
        subdiv=subdivision_code or None,
        years=range(start.year, end.year + 1),
    )
    return [
        (holiday_date, name)
        for holiday_date, name in sorted(calendar.items())
        if start <= holiday_date <= end
    ]
