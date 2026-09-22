"""Pipeline test only; does not encrypt or satisfy security requirements."""
import json
import numpy as np
import scipy.sparse as sp
from harness.model import predict


class Adapter:
    def describe(self):
        return {"name": "plaintext_debug", "is_fhe": False, "scheme": "none"}

    def keygen(self, threads):
        return {}, {}

    def encrypt(self, sensitive, private_files, threads):
        return {"debug.json": json.dumps(sensitive).encode()}

    def evaluate(self, encrypted, public, public_files, threads, intermediate_dir):
        x = np.zeros((public["node_count"], public["feature_count"]))
        x[:, public["sensitive_indices"]] = json.loads(encrypted["debug.json"])
        x[:, public["public_indices"]] = public["x_public"]
        g = public["adjacency"]
        a = sp.coo_matrix((g["values"], (g["rows"], g["cols"])), shape=(len(x), len(x))).tocsr()
        scores = predict(x, a, public["weights"], public["sensitive_indices"], public["activation"])
        return {"debug.json": json.dumps(scores.tolist()).encode()}

    def decrypt(self, encrypted_scores, private_files, threads):
        return json.loads(encrypted_scores["debug.json"])
