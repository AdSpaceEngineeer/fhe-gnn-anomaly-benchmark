"""Fixed benchmark policy. Submitters must not modify this file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_SECURITY_BITS = 128
DEFAULT_THREADS = 2
KEY_POLICY = "fresh_per_invocation_reused_across_repeats"
SENSITIVE_FIELDS = ["transfer_amount_z", "source_daily_total_amount_z", "prior_report_count_z"]
STAGES = ("keygen", "encrypt", "evaluate", "decrypt")
SEAL_TC128_MAX_BITS = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}
