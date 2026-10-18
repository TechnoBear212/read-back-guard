## Test split

| index | n_errors | detected | detection_rate | detection_ci | n_correct | false_alarms | false_alarm_rate | false_alarm_ci | false_accusations | missed | say_again | say_again_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1b human text, real + injected | 106 | 100 | 0.943 | (0.882, 0.974) | 82 | 17 | 0.207 | (0.134, 0.307) | 17 | 6 | 0 | 0.0 |
| human text, real pairs | 24 | 23 | 0.958 | (0.798, 0.993) | 82 | 17 | 0.207 | (0.134, 0.307) | 17 | 1 | 0 | 0.0 |
| Whisper text, real pairs | 24 | 24 | 1.0 | (0.862, 1.0) | 82 | 51 | 0.622 | (0.514, 0.719) | 51 | 0 | 0 | 0.0 |
| Whisper + gate, real pairs | 24 | 24 | 1.0 | (0.862, 1.0) | 82 | 56 | 0.683 | (0.576, 0.774) | 45 | 0 | 19 | 0.179 |

WER {'atco': 0.174, 'pilot': 0.385, 'all': 0.25}, critical-value error rate 0.161 (n=366)

## Attribution, human vs Whisper (test, real pairs)

| gold | language understanding | right on both | right only on Whisper text | speech recognition | total |
|---|---|---|---|---|---|
| correct readback (false alarm if wrong) | 11 | 25 | 6 | 40 | 82 |
| real error (miss if wrong) | 0 | 23 | 1 | 0 | 24 |
| total | 11 | 48 | 7 | 40 | 106 |

{'human_only_right': 40, 'asr_only_right': 7, 'p_value': 0.0}

## Attribution, human vs Whisper + gate

| gold | language understanding | right on both | right only on Whisper text | speech recognition | total |
|---|---|---|---|---|---|
| correct readback (false alarm if wrong) | 13 | 22 | 4 | 43 | 82 |
| real error (miss if wrong) | 0 | 23 | 1 | 0 | 24 |
| total | 13 | 45 | 5 | 43 | 106 |

{'human_only_right': 43, 'asr_only_right': 5, 'p_value': 0.0}

## Own recordings

| index | n_errors | detected | detection_rate | detection_ci | n_correct | false_alarms | false_alarm_rate | false_alarm_ci | false_accusations | missed | say_again | say_again_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| script text | 10 | 10 | 1.0 | (0.722, 1.0) | 10 | 0 | 0.0 | (-0.0, 0.278) | 0 | 0 | 0 | 0.0 |
| Whisper text | 10 | 10 | 1.0 | (0.722, 1.0) | 10 | 4 | 0.4 | (0.168, 0.687) | 4 | 0 | 0 | 0.0 |
| Whisper + gate | 10 | 10 | 1.0 | (0.722, 1.0) | 10 | 4 | 0.4 | (0.168, 0.687) | 4 | 0 | 0 | 0.0 |

| case_id | verdict | script | whisper | whisper_gate | reason |
|---|---|---|---|---|---|
| r01 | CORRECT | CORRECT | CORRECT | CORRECT | all items read back |
| r02 | WRONG_VALUE | WRONG_VALUE | WRONG_CALLSIGN | WRONG_CALLSIGN | callsign sunza 452 read back as lufthansa 452 |
| r03 | CORRECT | CORRECT | WRONG_CALLSIGN | WRONG_CALLSIGN | callsign liner 3ab read back as ryanair 3ab |
| r04 | MISSING_ITEM | MISSING_ITEM | MISSING_ITEM | MISSING_ITEM | frequency 127.125 not read back |
| r05 | CORRECT | CORRECT | CORRECT | CORRECT | all items read back |
| r06 | WRONG_CALLSIGN | WRONG_CALLSIGN | WRONG_CALLSIGN | WRONG_CALLSIGN | callsign visor 530 read back as vizir 538 |
| r07 | CORRECT | CORRECT | WRONG_CALLSIGN | WRONG_CALLSIGN | callsign str6ma read back as austrian 6ma |
| r08 | WRONG_VALUE | WRONG_VALUE | WRONG_VALUE | WRONG_VALUE | squawk 2351 read back as 2357 |
| r09 | CORRECT | CORRECT | CORRECT | CORRECT | all items read back |
| r10 | WRONG_VALUE | WRONG_VALUE | WRONG_CALLSIGN | WRONG_CALLSIGN | callsign liner 82k read back as ryanair 82k |
| r11 | MISSING_ITEM | MISSING_ITEM | MISSING_ITEM | MISSING_ITEM | speed 220 not read back |
| r12 | CORRECT | CORRECT | WRONG_CALLSIGN | WRONG_CALLSIGN | callsign smart wings 456 read back as smart wing 456 |
| r13 | WRONG_CALLSIGN | WRONG_CALLSIGN | WRONG_CALLSIGN | WRONG_CALLSIGN | callsign 38w read back as csa 883w |
| r14 | CORRECT | CORRECT | WRONG_CALLSIGN | WRONG_CALLSIGN | callsign visor 21l read back as vizor 21l |
| r15 | WRONG_VALUE | WRONG_VALUE | WRONG_VALUE | WRONG_VALUE | frequency 121.9 read back as 121.8 |
| r16 | MISSING_ITEM | MISSING_ITEM | MISSING_ITEM | MISSING_ITEM | runway 06 not read back |
| r17 | CORRECT | CORRECT | CORRECT | CORRECT | all items read back |
| r18 | WRONG_VALUE | WRONG_VALUE | WRONG_VALUE | WRONG_VALUE | level FL120 read back as FL150 |
| r19 | CORRECT | CORRECT | CORRECT | CORRECT | all items read back |
| r20 | CORRECT | CORRECT | CORRECT | CORRECT | all items read back |