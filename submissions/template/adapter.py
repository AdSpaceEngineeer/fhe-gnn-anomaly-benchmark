"""Copy this directory to submissions/<your_name>/ and implement Adapter."""
class Adapter:
    def describe(self):
        return {"name": "your_method", "is_fhe": True, "scheme": "your_scheme",
                "security": {"classical_bits": 128, "validator": "external_review",
                             "evidence": "REPLACE with estimator version, inputs, outputs and threat assumptions"},
                "parameters": {"REPLACE": "full cryptographic parameter set"},
                "encoding": "REPLACE", "packing": "REPLACE", "activation": "REPLACE"}

    def keygen(self, threads):
        """Return (private_files, public_files); each maps filenames to bytes."""
        raise NotImplementedError("Implement key generation; server files MUST exclude secret keys")

    def encrypt(self, sensitive, private_files, threads):
        """Encode and encrypt N x 3 numbers. Return {filename: bytes}."""
        raise NotImplementedError

    def evaluate(self, encrypted, public, public_files, threads, intermediate_dir):
        """Return encrypted scores as {filename: bytes}. No decryption here.

        public: weights, normalized adjacency (COO rows/cols/values), x_public,
        sensitive/public column indices, feature_count, node_count, activation.
        Optional persisted intermediate files belong in intermediate_dir.
        """
        raise NotImplementedError

    def decrypt(self, encrypted_scores, private_files, threads):
        """Decode into a list of N scores in published node order."""
        raise NotImplementedError
