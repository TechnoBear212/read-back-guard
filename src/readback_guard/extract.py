from readback_guard.cache import cache_get, cache_key, cache_put
from readback_guard.config import ROOT
from readback_guard.schemas import Clearance

PROMPT_VERSION = "v1"

PROMPT = """You extract the clearance from one air traffic control radio transmission at Prague airport.
It may be from the controller or from the pilot reading it back. The text is a lowercase transcript
with numbers spoken digit by digit, and it may be cut off at the start or the end.

callsign: the aircraft callsign as spoken, lowercase, with the airline word kept, numbers as digits
and phonetic letters as single letters. Examples: "csa five papa charlie" -> "csa 5pc",
"lufthansa four five two" -> "lufthansa 452", "hotel zulu mike sierra four" -> "hzms4".
Use null if no callsign is spoken. Do not complete a callsign that is cut off.


items: one entry for each of these values that is spoken:
- level: digits only. unit "FL" for a flight level ("flight level one two zero" -> "120"),
  "ft" for an altitude ("altitude three thousand" -> "3000"). qualifier "climb" or "descend" if said.
- heading: three digits ("heading zero nine zero" -> "090"). qualifier "left" or "right" if said.
- speed: knots as digits ("one sixty" -> "160"). qualifier "increase" or "reduce" if said.
- frequency: MHz with a decimal point ("one two one decimal nine" -> "121.9").
- squawk: four digits ("one four zero four" -> "1404").
- qnh: four digits ("qnh one zero one three" -> "1013").
- runway: the runway number, with L, R or C if said ("runway three one" -> "31").

Rules:
- Pilots often drop the keyword: "cleared to land three one" is runway 31, "reducing one sixty" is speed 160,
  "one two one nine" after a frequency change is frequency 121.9.
- Extract only what is actually said in this transmission. Never fill in a value from what the other
  side probably said. If a value is cut off or only partly spoken, leave it out.
- Ignore wind, times, information letters, distances and taxiway names.

Transmission: {text}"""


def extract(text, cfg):
    from readback_guard.llm import generate_json

    cache_dir = ROOT / cfg["llm"]["cache_dir"]
    key = cache_key(cfg["llm"]["model"], PROMPT_VERSION, text)
    raw = cache_get(cache_dir, key)
    if raw is None:
        raw = generate_json(PROMPT.format(text=text), Clearance.model_json_schema(), cfg)
        Clearance.model_validate_json(raw)
        cache_put(cache_dir, key, raw)
    return Clearance.model_validate_json(raw)
