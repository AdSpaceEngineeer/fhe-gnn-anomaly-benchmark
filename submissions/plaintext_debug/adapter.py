"""Non-FHE adapter for end-to-end harness verification on either workload."""
import json
import numpy as np
from harness.model import predict


class Adapter:
    def configure(self, workload, threads):
        self.workload = workload
        self.threads = threads

    def describe(self):
        return dict(name='plaintext_debug', is_fhe=False, scheme='none', supported_workloads=['gcn', 'tam'])

    def thread_report(self, stage, requested_threads):
        return dict(compute_threads=self.threads, worker_processes=1,
                    threading_model='NumPy/SciPy; BLAS thread limit set by harness environment')

    def keygen(self, threads):
        return {}, {}

    def encrypt(self, sensitive, private_files, threads):
        return {'debug.json': json.dumps(sensitive).encode()}

    def evaluate(self, encrypted, public, public_files, threads, intermediate_dir):
        x = np.zeros((public['node_count'], public['feature_count']))
        x[:, public['sensitive_indices']] = json.loads(encrypted['debug.json'])
        x[:, public['public_indices']] = public['x_public']
        return {'debug.json': json.dumps(predict(x, public).tolist()).encode()}

    def decrypt(self, encrypted_scores, private_files, threads):
        return json.loads(encrypted_scores['debug.json'])
