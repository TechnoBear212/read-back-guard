from readback_guard.normalize import callsigns_match, norm_value
from readback_guard.schemas import SCORED_TYPES, ItemResult, Verdict


def compare(atco, pilot):
    """Check every scored controller item against the pilot's items and return one Verdict.

    Precedence: WRONG_CALLSIGN, then WRONG_VALUE, then MISSING_ITEM, then CORRECT.
    """
    results = []
    for a in atco.items:
           
        want = norm_value(a.type, a.value, a.unit)
        if a.type not in SCORED_TYPES or want == "":
            continue
        got = [norm_value(p.type, p.value, p.unit) for p in pilot.items if p.type == a.type]
        got = [g for g in got if g != ""]
        if not got:
        
            results.append(ItemResult(type=a.type, atco_value=want, pilot_value=None, result="MISSING"))
        elif want in got:
            results.append(ItemResult(type=a.type, atco_value=want, pilot_value=want, result="OK"))
        else:
            results.append(ItemResult(type=a.type, atco_value=want, pilot_value=got[0], result="WRONG"))

    if not callsigns_match(atco.callsign, pilot.callsign):
        return Verdict(verdict="WRONG_CALLSIGN", reason=f"callsign {atco.callsign} read back as {pilot.callsign}", items=results)
    for r in results:
        if r.result == "WRONG":
            return Verdict(verdict="WRONG_VALUE", reason=f"{r.type} {r.atco_value} read back as {r.pilot_value}", items=results)
    for r in results:
        if r.result == "MISSING":
            return Verdict(verdict="MISSING_ITEM", reason=f"{r.type} {r.atco_value} not read back", items=results)
    return Verdict(verdict="CORRECT", reason="all items read back", items=results)
