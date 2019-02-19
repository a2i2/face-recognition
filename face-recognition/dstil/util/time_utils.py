import datetime
import pytz
from tzlocal import get_localzone


def timestamp_local():
    """
    Returns a timestamp in the local timezone. If the TZ environment variable has not been set
    and the local timezone cannot be found, this method will return the time in UTC.
    """
    local_tz = get_localzone()
    return datetime.datetime.now(tz=pytz.utc).replace(tzinfo=pytz.utc).astimezone(local_tz)


def timestamp_local_iso_format():
    """
    Returns a local timestamp as an ISO8601-formatted string.
    """
    return timestamp_local().isoformat()


def timestamp_local_path_format():
    """
    Returns a local timestamp in a format that can be used cross platform in a file path.
    """
    return timestamp_local().strftime('%Y-%m-%d-%H-%M-%S')
