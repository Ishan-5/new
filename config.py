import os
from dotenv import load_dotenv

load_dotenv()

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL          = "llama-3.3-70b-versatile"   # cheap + fast for eval runs
MAX_TOKENS     = 1024
TEMPERATURE    = 1.0   # keep at 1.0 so repeated runs are actually different

# ── Eval settings ─────────────────────────────────────────────────────────────
RUNS_PER_TASK  = 5     # how many times to run each task (more = better stats)
MAX_STEPS      = 6     # max tool calls per agent run before we force-stop

# ── Pricing (USD per million tokens) ─────────────────────────────────────────
COST_PER_1M_INPUT_TOKENS  = 0.59
COST_PER_1M_OUTPUT_TOKENS = 0.79

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH        = "database/results.db"
TASKS_PATH     = "benchmarks/tasks.json"
RUBRICS_PATH   = "benchmarks/rubrics.json"
REPORTS_DIR    = "reports"
SAMPLE_FILES   = "sample_files"

# ── API key ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not set. Add it to a .env file:\n"
        "  GROQ_API_KEY=gsk_..."
    )