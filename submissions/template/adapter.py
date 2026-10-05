"""Copy this directory into submissions/<method>/ and implement Adapter."""


class Adapter:
    def configure(self, workload, threads):
        """Called in every fresh stage process; configure the actual backend."""
        self.workload, self.threads = workload, threads

    def describe(self):
        return dict(name='your_method', supported_workloads=['gcn', 'tam'],
                    is_fhe=True, scheme='REPLACE',
                    security=dict(classical_bits=128, validator='external_review',
                                  evidence='REPLACE: estimator, version, inputs, outputs and assumptions'),
                    parameters={'REPLACE': 'cryptographic parameter set'},
                    encoding='REPLACE', packing='REPLACE', activation='REPLACE: encrypted nonlinear implementation')

    def thread_report(self, stage, requested_threads):
        """Report the configuration actually used, not merely the requested limit.

        compute_threads: maximum configured concurrent compute threads, summed
        across worker processes. Describe independent native pools/nesting.
        The harness separately samples OS threads, which include idle/I/O threads.
        """
        raise NotImplementedError('Return compute_threads, worker_processes and threading_model')

    def keygen(self, threads):
        """Return (private_files, public_files), each {filename: bytes}."""
        raise NotImplementedError('Server files must exclude secret keys')

    def encrypt(self, sensitive, private_files, threads):
        """Encode/encrypt the N x 3 sensitive matrix. Return serialized files."""
        raise NotImplementedError

    def evaluate(self, encrypted, public, public_files, threads, intermediate_dir):
        """Return encrypted scores in published node order. No secret keys here.

        public contains workload, weights, adjacency, public feature columns,
        sensitive/public indices, feature_count, node_count and activation.
        TAM also receives scoring_adjacency and norm_epsilon.
        Optional intermediate_dir/server_reported_steps.json records seconds:
        {"Encrypted computation": 1.2, "I/O": 0.3, "Total": 1.5}.
        """
        raise NotImplementedError

    def decrypt(self, encrypted_scores, private_files, threads):
        """Decode and return a list of N scores; do not refit the threshold."""
        raise NotImplementedError
