MODEL = "gpt-6-luna"
PROMPT_VERSION = "v2"
MAX_LLM_ITERATIONS = 5 # Max model calls within ONE user turn
MAX_INPUT_CHARS = 2_000
MAX_OUTPUT_TOKENS = 2_000 # need to enlarge if enable model reasoning 
# Max user turns sent to the model as history, including the current one.
# A turn = one user message plus its tool calls, tool results and the reply;
# older turns are dropped whole (still shown in the UI and saved to the DB).
MAX_HISTORY_TURNS = 8
# Max messages one user may send in any rolling 24 hours. 
MAX_DAILY_MESSAGES = 50
