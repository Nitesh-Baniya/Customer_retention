# Failure diagnosis from traces

> OFFLINE STUB RUN - illustrates the workflow; the agent here is a scripted policy.


## prompt v1 - 5 / 8 cases with a failure mode


### `answered_from_memory` (2 case(s))

Diagnosis: Expected one of ['web_search'] but the agent used no tool; it answered from parametric knowledge instead of calling a tool.

Cases: population_compare, renewables_latest

Example trace `population_compare` (termination: success_no_verification_needed):

  1. **draft_answer** - reasoning: Model returned no tool call -> loop proposes to stop.
     result: `Brazil has roughly 215 million people and India about 1.4 billion.`
  2. **verification_decision** - reasoning: No draft answer was provided to evaluate.
  3. **final_answer** - reasoning: confidence=low
     result: `Brazil has roughly 215 million people and India about 1.4 billion.`


### `verifier_skipped` (3 case(s))

Diagnosis: Verification was expected but not run. Verifier said: No draft answer was provided to evaluate.

Cases: weather_tokyo, weather_compare, weather_paris

Example trace `weather_tokyo` (termination: success_no_verification_needed):

  1. **tool_call** `get_current_weather`({"location": "Tokyo"}) - reasoning: I need `get_current_weather` for this question.
     result: `{"location": "Tokyo", "condition": "Mainly clear", "temperature": 22.1, "temperature_unit": "\u00b0C", "relati`
  2. **draft_answer** - reasoning: Model returned no tool call -> loop proposes to stop.
     result: `Tokyo: 22.1°C, Mainly clear.`
  3. **verification_decision** - reasoning: No draft answer was provided to evaluate.
  4. **final_answer** - reasoning: confidence=medium
     result: `Tokyo: 22.1°C, Mainly clear.`


## prompt v2 - 5 / 8 cases with a failure mode


### `verifier_skipped` (5 case(s))

Diagnosis: Verification was expected but not run. Verifier said: No draft answer was provided to evaluate.

Cases: weather_tokyo, weather_compare, population_compare, renewables_latest, weather_paris

Example trace `weather_tokyo` (termination: success_no_verification_needed):

  1. **tool_call** `get_current_weather`({"location": "Tokyo"}) - reasoning: I need `get_current_weather` for this question.
     result: `{"location": "Tokyo", "condition": "Mainly clear", "temperature": 22.1, "temperature_unit": "\u00b0C", "relati`
  2. **draft_answer** - reasoning: Model returned no tool call -> loop proposes to stop.
     result: `Tokyo: 22.1°C, Mainly clear.`
  3. **verification_decision** - reasoning: No draft answer was provided to evaluate.
  4. **final_answer** - reasoning: confidence=medium
     result: `Tokyo: 22.1°C, Mainly clear.`


## prompt v3 - 0 / 8 cases with a failure mode

No failures.
