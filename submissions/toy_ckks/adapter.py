"""Incomplete GCN CKKS example. The encrypted ReLU method must be supplied."""
import argparse
import json
import sys


class UnsupportedOperationError(NotImplementedError):
    pass


RELU_MESSAGE = (
    'Unsupported operation: encrypted ReLU. Replace the encrypted_relu() '
    'placeholder in this submission with your encrypted activation method. '
    'Do not decrypt inside the evaluator.'
)


def backend():
    # Metadata and the readiness check do not require the optional FHE package.
    import tenseal
    return tenseal


def add_terms(terms):
    iterator = iter(terms)
    total = next(iterator)
    for term in iterator:
        total = total + term
    return total


class Adapter:
    # Set True only after implementing encrypted_relu and reviewing the full
    # circuit's parameter requirements. This flag supplies no activation method.
    activation_ready = False
    parameters = dict(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    scale = 2 ** 40

    def configure(self, workload, threads):
        if workload != 'gcn':
            raise ValueError('This CKKS example supports GCN only')
        if type(threads) is not int or threads < 1:
            raise ValueError('threads must be a positive integer')
        self.threads = threads

    def describe(self):
        return dict(name='toy_ckks', supported_workloads=['gcn'], is_fhe=True,
                    scheme='CKKS', parameters=self.parameters,
                    security=dict(classical_bits=128, validator='seal_tc128',
                                  evidence='Microsoft SEAL default tc128 validation; '
                                  '8192-degree context with 200 total coefficient-modulus bits'),
                    encoding='Real-valued CKKS; global scale ' + str(self.scale),
                    packing='One scalar per CKKS vector; no cross-node packing',
                    activation='Submitter-defined encrypted ReLU; placeholder in this example',
                    unsupported_operations=[] if self.activation_ready else [RELU_MESSAGE])

    def thread_report(self, stage, requested_threads):
        return dict(compute_threads=self.threads, worker_processes=1,
                    threading_model='Sequential Python loops; each TenSEAL context uses '
                    'the configured n_threads; no submission worker processes')

    def keygen(self, threads):
        ts = backend()
        context = ts.context(ts.SCHEME_TYPE.CKKS, n_threads=threads, **self.parameters)
        context.global_scale = self.scale
        context.generate_relin_keys()
        private = context.serialize(save_public_key=True, save_secret_key=True,
                                    save_galois_keys=False, save_relin_keys=False)
        public = context.serialize(save_public_key=True, save_secret_key=False,
                                   save_galois_keys=False, save_relin_keys=True)
        return {'context.bin': private}, {'context.bin': public}

    def encrypt(self, sensitive, private_files, threads):
        ts = backend()
        context = ts.context_from(private_files['context.bin'], n_threads=threads)
        if any(len(row) != 3 for row in sensitive):
            raise ValueError('Expected three sensitive features per node')
        return {'input_%08d_%d.bin' % (i, j): ts.ckks_vector(context, [float(value)]).serialize()
                for i, row in enumerate(sensitive) for j, value in enumerate(row)}

    def encrypted_relu(self, ciphertext, public_context):
        """REPLACE THIS PLACEHOLDER: return an encrypted activation result.

        Input/output: a one-element CKKS vector under the evaluation key set.
        Only a public context is available. Return a compatible ciphertext for
        subsequent GCN arithmetic. The reference function is max(z, 0).
        No polynomial, lookup table, identity fallback or client interaction is
        implemented here. Describe and validate the method you supply.
        """
        raise UnsupportedOperationError(RELU_MESSAGE)

    def evaluate(self, encrypted, public, public_files, threads, intermediate_dir):
        if public['workload'] != 'gcn' or public['activation'] != 'relu':
            raise ValueError('Expected the frozen ReLU GCN workload')
        ts = backend()
        context = ts.context_from(public_files['context.bin'], n_threads=threads)
        if context.has_secret_key():
            raise ValueError('Evaluator context must not contain secret keys')
        n = public['node_count']
        sensitive = public['sensitive_indices']
        original = [[ts.ckks_vector_from(context, encrypted['input_%08d_%d.bin' % (i, j)])
                     for j in range(3)] for i in range(n)]
        hidden = original
        graph = public['adjacency']
        neighbours = [[] for _ in range(n)]
        for row, col, value in zip(graph['rows'], graph['cols'], graph['values']):
            neighbours[row].append((col, value))
        layers = ('encoder_1', 'encoder_2', 'decoder_1', 'decoder_2')
        for layer_index, name in enumerate(layers):
            weights, bias = public['weights'][name + '.weight'], public['weights'][name + '.bias']
            projected = []
            for i in range(n):
                row = []
                for j in range(len(bias)):
                    if layer_index == 0:
                        value = add_terms(hidden[i][k] * weights[column][j]
                                          for k, column in enumerate(sensitive))
                        value = value + sum(public['x_public'][i][k] * weights[column][j]
                                            for k, column in enumerate(public['public_indices']))
                    else:
                        value = add_terms(hidden[i][k] * weights[k][j] for k in range(len(hidden[i])))
                    row.append(value)
                projected.append(row)
            hidden = [[add_terms(projected[col][j] * coefficient for col, coefficient in neighbours[i]) + bias[j]
                       for j in range(len(bias))] for i in range(n)]
            if layer_index < 3:
                hidden = [[self.encrypted_relu(value, context) for value in row] for row in hidden]
        results = {}
        for i in range(n):
            errors = [hidden[i][column] - original[i][k] for k, column in enumerate(sensitive)]
            score = add_terms(error * error for error in errors) * (1.0 / len(sensitive))
            results['score_%08d.bin' % i] = score.serialize()
        return results

    def decrypt(self, encrypted_scores, private_files, threads):
        ts = backend()
        context = ts.context_from(private_files['context.bin'], n_threads=threads)
        return [float(ts.ckks_vector_from(context, encrypted_scores[name]).decrypt()[0])
                for name in sorted(encrypted_scores)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='Run a three-value crypto round trip, not GCN inference')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    adapter = Adapter()
    adapter.configure('gcn', args.threads)
    try:
        if not args.demo:
            if not adapter.activation_ready:
                raise UnsupportedOperationError(RELU_MESSAGE)
            print('Activation declared implemented; run the benchmark to verify the submission.')
            return 0
        private, public = adapter.keygen(args.threads)
        sample = [[0.75, 1.20, 0.40]]
        encrypted = adapter.encrypt(sample, private, args.threads)
        # Round-trip input values, not GCN scores. No key material is printed/saved.
        decoded = adapter.decrypt(encrypted, private, args.threads)
        print(json.dumps(dict(demo='Encryption/decryption only; not a benchmark result',
                              plaintext=sample[0], decrypted=decoded), indent=2), flush=True)
        ts = backend()
        context = ts.context_from(public['context.bin'], n_threads=args.threads)
        first = ts.ckks_vector_from(context, encrypted['input_00000000_0.bin'])
        adapter.encrypted_relu(first, context)
        return 0
    except UnsupportedOperationError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
