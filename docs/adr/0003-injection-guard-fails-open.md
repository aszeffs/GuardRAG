# The prompt-injection classifier fails open

Input screening uses cheap heuristics plus Meta's Llama Prompt Guard on Groq. Prompt Guard is a preview model that may go down or be retired. When it is unavailable, requests continue with heuristics only, and a warning is logged. Failing closed would take the whole assistant down because of one optional layer. The layers after it (passage-only citations, refusal without support, PII redaction) still limit what a missed injection can do.
