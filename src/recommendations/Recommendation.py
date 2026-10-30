def classify(tmax, tmin, rain):
    flags = []
    if tmax >= 45: flags.append("severe_heat")
    elif tmax >= 40: flags.append("heat")
    if tmin <= 10: flags.append("cold")
    if rain >= 204.4: flags.append("extreme_rain")
    elif rain >= 115.6: flags.append("very_heavy_rain")
    elif rain >= 64.5: flags.append("heavy_rain")
    elif rain < 2.5: flags.append("dry")
    return flags
 
RULES = {
 "severe_heat":     ["Issue heat-wave alert; restrict outdoor work 11:00-16:00", "Prepare cooling centres and ORS stock"],
 "heat":            ["Advise irrigation in early morning/evening", "Public advisory: hydrate, avoid midday exposure"],
 "cold":            ["Protect sensitive crops (frost cover/light irrigation)"],
 "heavy_rain":      ["Clear drains; alert low-lying areas", "Delay pesticide/fertiliser application"],
 "very_heavy_rain": ["Activate flood-watch; check dam/river levels"],
 "extreme_rain":    ["Issue red alert; pre-position rescue teams"],
 "dry":             ["Monitor soil moisture; plan water-saving irrigation"],
}
 
def recommend(district, tmax, tmin, rain, drivers=None):
    flags = classify(tmax, tmin, rain)
    recs = [r for f in flags for r in RULES[f]] or ["No action needed; continue monitoring"]
    why = ""
    if drivers:
        why = "Main model drivers: " + ", ".join(f"{n} ({v:+.2f})" for n, v in drivers)
    return dict(district=district, flags=flags, actions=recs, explanation=why)