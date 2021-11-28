from typing import Dict, Optional

import numpy as np


def summarize(recordings: Dict) -> Dict:
    heart_rate_series = recordings.get("heart_rate")
    heart_rate: Optional[int]
    if heart_rate_series is not None:
        heart_rate = round(np.nanmedian(heart_rate_series))
    else:
        heart_rate = None

    stride_rate_series = recordings.get("cadence")
    step_rate: Optional[int]
    if stride_rate_series is not None:
        stride_rate = round(np.nanmedian(stride_rate_series))
        step_rate = 2 * stride_rate
    else:
        step_rate = None

    timestamps = recordings["timestamp"]
    time_start = timestamps[0]
    time_end = timestamps[-1]
    duration_s = time_end - time_start
    duration = duration_s.item()
    distance = recordings["distance"][-1]

    speed_series = recordings.get("speed")
    speed: Optional[float] = None
    if speed_series is not None:
        speed_m_per_sec = np.nanmedian(speed_series)
        speed = speed_m_per_sec * 1e-3 * 3600

    altitude_series = recordings.get("altitude")
    ascent: Optional[float] = None
    descent: Optional[float] = None
    if altitude_series is not None:
        diff = np.diff(altitude_series)
        mask_ascent = diff > 0
        ascent = round(diff[mask_ascent].sum())
        mask_descent = diff < 0
        descent = round(diff[mask_descent].sum())

    return dict(
        distance=distance,
        duration=duration,
        speed=speed,
        ascent=ascent,
        descent=descent,
        heart_rate=heart_rate,
        step_rate=step_rate,
    )
