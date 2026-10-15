from readback_guard.compare import compare
from readback_guard.extract import extract


def check_pair(atco_text, pilot_text, cfg):
    atco = extract(atco_text, cfg)
    pilot = extract(pilot_text, cfg)
    return atco, pilot, compare(atco, pilot)
