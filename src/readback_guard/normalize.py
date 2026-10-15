import re


def norm_callsign(callsign):
    """Lowercase, keep only letters and digits. None becomes ""."""
    if callsign is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", callsign.lower())


def callsigns_match(atco, pilot):
    """True if either side is empty, or one normalised callsign starts or ends with the other."""
    a = norm_callsign(atco)
    p = norm_callsign(pilot)
    if a == "" or p == "":
        return True
    return a.startswith(p) or p.startswith(a) or a.endswith(p) or p.endswith(a)


def norm_value(kind, value, unit=None):
    """One canonical string per value, e.g. FL120, A3000, 090, 160, 121.9, 1404, 24L, 1013."""

    v = value.upper().replace("RWY", "").replace(" ", "")
    if not re.search(r"[0-9]", v):
        return ""
    if kind == "level":
        prefix = "A" if unit == "ft" else "FL"
        return prefix + str(int(v))
    if kind == "heading":
        return str(int(v)).zfill(3)
    if kind == "squawk":
        return v.zfill(4)
    if kind == "frequency":
        return f"{float(v):.3f}".rstrip("0").rstrip(".")
    if kind == "runway":
        number = re.sub(r"[^0-9]", "", v)
        side = re.sub(r"[^LRC]", "", v)
        return number.zfill(2) + side
    return str(int(v))