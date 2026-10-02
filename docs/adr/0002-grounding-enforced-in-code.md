# Grounding is enforced in code, not trusted to the prompt

The model may cite only passages it was given in this request. Citations to anything else are dropped after generation, and an answer left with no valid Citation is turned into an out-of-corpus Refusal. Confidence (`high`/`low`/`none`) is computed from retrieval and citation evidence, never taken from the model. A prompt instruction to "only answer from the context" can be bypassed by injection or ignored by the model; a post-generation check can't. An LLM's opinion of its own confidence is uncalibrated and would not hold up as a metric.

## Consequences

Some correct answers will be refused when the model cites badly. That shows up as a lower answer rate rather than as hallucinations, which is the trade-off we want for a government-services assistant.
