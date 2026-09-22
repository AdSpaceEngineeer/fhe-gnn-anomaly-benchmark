"""Readable scalar CKKS implementation of all four GCN layers, for tiny graphs."""
import json
import numpy as np
import tenseal as ts
from harness.model import LAYERS

DEGREE = 32768
CHAIN = [60] + [40] * 12 + [60]  # 600 total bits <= SEAL tc128 limit 881
SCALE = 2 ** 40


def sum_ciphertexts(values):
    if not values:
        raise ValueError("Empty encrypted sum")
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


class Adapter:
    def describe(self):
        return {"name": "toy_ckks", "is_fhe": True, "scheme": "CKKS", "backend": "TenSEAL 0.3.16 / SEAL",
                "security": {"classical_bits": 128, "validator": "seal_tc128",
                             "evidence": "SEAL default tc128 validation; ternary secret, standard SEAL error distribution; https://github.com/microsoft/SEAL/blob/v4.1.2/native/src/seal/util/hestdparms.h"},
                "parameters": {"poly_modulus_degree": DEGREE, "coeff_mod_bit_sizes": CHAIN, "scale_bits": 40},
                "encoding": "CKKS real scalars after frozen client preprocessing",
                "packing": "one scalar per ciphertext, deliberately unoptimized",
                "activation": "exact circuit z + 0.125*z*z, approximate CKKS arithmetic",
                "limitations": "At most 8 event nodes; no bootstrapping; not intended for full-scale inference"}

    def keygen(self, threads):
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=DEGREE,
                         coeff_mod_bit_sizes=CHAIN, n_threads=threads)
        ctx.global_scale = SCALE
        ctx.generate_relin_keys()
        # No slot reductions or rotations: no Galois keys needed.
        private = ctx.serialize(save_public_key=True, save_secret_key=True,
                                save_galois_keys=False, save_relin_keys=False)
        public = ctx.serialize(save_public_key=True, save_secret_key=False,
                               save_galois_keys=False, save_relin_keys=True)
        return {"context.bin": private}, {"context.bin": public}

    def encrypt(self, sensitive, private_files, threads):
        if len(sensitive) > 8:
            raise ValueError("toy_ckks supports at most 8 nodes; use a scalable submission for the full bundle")
        ctx = ts.context_from(private_files["context.bin"], n_threads=threads)
        return {"%d_%d.bin" % (i, j): ts.ckks_vector(ctx, [float(v)]).serialize()
                for i, row in enumerate(sensitive) for j, v in enumerate(row)}

    def evaluate(self, encrypted, public, public_files, threads, intermediate_dir):
        if public["activation"] != "poly2":
            raise ValueError("toy_ckks implements only the frozen poly2 activation")
        ctx = ts.context_from(public_files["context.bin"], n_threads=threads)
        if ctx.has_secret_key():
            raise ValueError("Evaluator must not receive secret keys")
        n = public["node_count"]
        if n > 8:
            raise ValueError("toy_ckks supports at most 8 nodes")
        xs = [[ts.ckks_vector_from(ctx, encrypted["%d_%d.bin" % (i, j)]) for j in range(3)] for i in range(n)]
        g = public["adjacency"]
        neighbors = [[] for _ in range(n)]
        for row, col, value in zip(g["rows"], g["cols"], g["values"]):
            neighbors[row].append((col, float(value)))
        h = xs
        weights = public["weights"]
        for layer_index, name in enumerate(LAYERS):
            w = np.asarray(weights[name + ".weight"])
            bias = weights[name + ".bias"]
            next_h = []
            for i in range(n):
                row = []
                for out in range(w.shape[1]):
                    terms = []
                    plain = float(bias[out])
                    for j, coefficient in neighbors[i]:
                        indices = public["sensitive_indices"] if layer_index == 0 else range(w.shape[0])
                        for local, feature in enumerate(indices):
                            scalar = coefficient * float(w[feature, out])
                            if scalar != 0:
                                terms.append(h[j][local] * scalar)
                        if layer_index == 0:
                            plain += coefficient * float(np.dot(public["x_public"][j], w[public["public_indices"], out]))
                    # Fuse graph and feature weights to use one multiplicative level.
                    # An all-zero channel is a public constant; encrypt it at the server.
                    z = (sum_ciphertexts(terms) + plain) if terms else ts.ckks_vector(ctx, [plain])
                    if layer_index < 3:
                        z = z + z.square() * 0.125
                    row.append(z)
                next_h.append(row)
            h = next_h
        scores = {}
        for i in range(n):
            errors = [(h[i][feature] - xs[i][j]).square() for j, feature in enumerate(public["sensitive_indices"])]
            scores["%d.bin" % i] = (sum_ciphertexts(errors) * (1.0 / 3.0)).serialize()
        return scores

    def decrypt(self, encrypted_scores, private_files, threads):
        ctx = ts.context_from(private_files["context.bin"], n_threads=threads)
        return [ts.ckks_vector_from(ctx, encrypted_scores["%d.bin" % i]).decrypt()[0]
                for i in range(len(encrypted_scores))]
