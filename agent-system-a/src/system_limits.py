

TOP_LEVEL_RECURSION_LIMIT = 12
RAG_AGENT_RECURSION_LIMIT = 6
# One reasoning call, one tool action, and one final observation are enough for
# web research because the separate Answer Agent owns cross-source synthesis.
WEB_AGENT_RECURSION_LIMIT = 4
VISUAL_AGENT_RECURSION_LIMIT = 6
MANUAL_VISUAL_AGENT_RECURSION_LIMIT = 6
ANSWER_AGENT_RECURSION_LIMIT = 8
# Two distinct MCP tools (protocol lookup, severity calculator) may both be
# needed in one turn, so this budget is slightly larger than the single-tool
# specialists above.
PROTOCOL_AGENT_RECURSION_LIMIT = 6

GENERATION_VALIDATION_ATTEMPTS = 3
OUTPUT_REPAIR_ATTEMPTS = 1
PROVIDER_FALLBACK_ATTEMPTS = 1
TOOL_RETRY_ATTEMPTS = 1

RAG_RETRIEVAL_MAX_CALLS = 1
WEB_SEARCH_MAX_CALLS = 2
IMAGE_INSPECTION_MAX_CALLS = 1
MANUAL_VISUAL_MAX_CALLS = 1
MCP_TOOL_MAX_CALLS = 2
