"""Workload registry and common benchmark policy."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_SECURITY_BITS = 128
DEFAULT_THREADS = 2
KEY_POLICY = 'fresh_per_invocation_reused_across_repeats'
STAGES = ('keygen', 'encrypt', 'evaluate', 'decrypt')
WORKLOADS = {
    'gcn': {'artifact_id': 'scam-list-gcn-relu-100k-v2', 'name': 'Scam_List_GCN',
            'activation': 'relu', 'features': 8, 'node_unit': 'events',
            'sensitive': ['transfer_amount_z', 'source_daily_total_amount_z', 'prior_report_count_z']},
    'tam': {'artifact_id': 'scam-list-tam-synthetic-v1', 'name': 'Scam_List_TAM',
            'activation': 'prelu', 'features': 10, 'node_unit': 'accounts',
            'sensitive': ['account_age_days_z', 'interaction_count_30d_z', 'cross_border_partner_fraction_30d_z']},
}
SEAL_TC128_MAX_BITS = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}
