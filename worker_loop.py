"""
Dedicated worker service — runs continuously, polling every 30 min.
Deploy as a separate Railway service so it never misses a cron window.
Only active between 9am–11:30pm ET.
"""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from eastern_time import today_et

POLL_INTERVAL = 30 * 60  # 30 minutes


def in_active_window() -> bool:
    """Only run during game hours: 9am–11:30pm ET."""
    now = datetime.now()
    # Use UTC offset: ET = UTC-4 (EDT) or UTC-5 (EST)
    try:
        import pytz
        et = pytz.timezone('America/New_York')
        now_et = datetime.now(et)
        hour = now_et.hour + now_et.minute / 60
    except Exception:
        # Rough fallback: UTC-4
        hour = (now.hour - 4) % 24 + now.minute / 60
    return 9.0 <= hour <= 23.5


if __name__ == '__main__':
    print('Worker loop started.')
    while True:
        if in_active_window():
            try:
                from worker import run
                run()
            except Exception as e:
                print(f'Worker error: {e}')
        else:
            print(f'Outside active window — sleeping.')
        time.sleep(POLL_INTERVAL)
