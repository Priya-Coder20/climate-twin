"""Effect/impact module + what-if scenarios."""
import numpy as np
 
def heat_stress_index(tmax, tmin):
    return float(np.clip((0.6*tmax + 0.4*tmin - 28) / 12, 0, 1))   # 0..1 (simple proxy)
 
def crop_stress(tmax, rain_7d_mm):
    heat = np.clip((tmax - 35) / 10, 0, 1)
    water = np.clip((30 - rain_7d_mm) / 30, 0, 1)
    return float(0.5*heat + 0.5*water)
 
def flood_risk(rain_3d_mm):
    return float(np.clip(rain_3d_mm / 200, 0, 1))
 
def impact_summary(tmax, tmin, rain_3d, rain_7d):
    return dict(heat_stress=heat_stress_index(tmax, tmin),
                crop_stress=crop_stress(tmax, rain_7d),
                flood_risk=flood_risk(rain_3d))
 
def what_if(predict_fn, X_row, feature_names, temp_delta=0.0, rain_factor=1.0):
    """Perturb inputs (e.g. +2 C, rain x0.8) and re-predict. Required by the problem statement."""
    X2 = X_row.copy()
    for i, n in enumerate(feature_names):
        if "tmax" in n or "tmin" in n or "temp" in n: X2[:, i] += temp_delta
        if "rain" in n: X2[:, i] *= rain_factor
    return float(predict_fn(X_row)[0]), float(predict_fn(X2)[0])
 