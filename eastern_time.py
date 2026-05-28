"""
Eastern Time helpers — ensures dates stay correct on UTC servers.
Always use these instead of datetime.now() for display/date logic.
"""
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
    def now_et() -> datetime:
        return datetime.now(_ET)
except ImportError:
    def now_et() -> datetime:
        from datetime import timezone, timedelta
        return datetime.now(timezone(timedelta(hours=-5)))


def today_et() -> 'datetime.date':
    return now_et().date()


def today_str_et() -> str:
    return now_et().strftime('%Y-%m-%d')
