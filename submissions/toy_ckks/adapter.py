"""Copy this directory to submissions/my_method: the same frozen GCN, real CKKS."""
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import tenseal.sealapi as seal
from packed_ckks import (DEGREE, CHAIN, SCALE, STRIDE, SLOTS, ROWS, SENSITIVE, LAYERS,
                         Files, Arithmetic, parameters, context)


class Adapter:
    def describe(self):
        return {"name": Path(__file__).resolve().parent.name, "is_fhe": True, "scheme": "CKKS",
                "backend": "TenSEAL 0.3.16 native Microsoft SEAL bindings",
                "security": {"classical_bits": 128, "validator": "seal_tc128_native",
                             "evidence": "Explicit SEAL TC128 context; standard ternary secret and error sampling; https://github.com/microsoft/SEAL/blob/v4.1.2/native/src/seal/util/hestdparms.h"},
                "parameters": {"poly_modulus_degree": DEGREE, "coeff_mod_bit_sizes": CHAIN, "scale_bits": 40},
                "encoding": "Frozen normalized sensitive columns, CKKS real values",
                "packing": "256 event rows per ciphertext, 64 slots per row; public sparse graph unchanged",
                "activation": "z + 0.125*z*z, approximate CKKS arithmetic",
                "limitations": "Readable correctness example; no bootstrapping or performance claims; large graph inference can be very slow"}

    def keygen(self, threads):
        with tempfile.TemporaryDirectory(prefix="scam-ckks-keygen-") as temporary:
            files = Files(temporary)
            p = parameters()
            ctx = context(p)
            keygen = seal.KeyGenerator(ctx)
            public, relin, galois = seal.PublicKey(), seal.RelinKeys(), seal.GaloisKeys()
            keygen.create_public_key(public)
            keygen.create_relin_keys(relin)
            elements = [pow(3, 1 << bit, 2 * DEGREE) for bit in range(SLOTS.bit_length() - 1)]
            keygen.create_galois_keys(elements, galois)
            encoded_parameters, encoded_public = files.dump(p), files.dump(public)
            private = {"parameters.bin": encoded_parameters, "public.key": encoded_public,
                       "secret.key": files.dump(keygen.secret_key())}
            published = {"parameters.bin": encoded_parameters, "public.key": encoded_public,
                         "relin.key": files.dump(relin), "galois.key": files.dump(galois)}
            return private, published

    def encrypt(self, sensitive, private_files, threads):
        values = np.asarray(sensitive, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("Expected finite N x 3 sensitive features")
        result = {"shape.json": json.dumps({"nodes": len(values), "stride": STRIDE, "rows": ROWS}).encode()}
        with tempfile.TemporaryDirectory(prefix="scam-ckks-encrypt-") as temporary:
            files = Files(temporary)
            ctx = context(files.load(seal.EncryptionParameters, private_files["parameters.bin"]))
            public_key = files.load(seal.PublicKey, private_files["public.key"], ctx)
            encoder, encryptor = seal.CKKSEncoder(ctx), seal.Encryptor(ctx, public_key)
            for block, start in enumerate(range(0, len(values), ROWS)):
                count = min(ROWS, len(values) - start)
                slots = np.zeros((ROWS, STRIDE))
                slots[:count, SENSITIVE] = values[start:start + count]
                plain, cipher = seal.Plaintext(), seal.Ciphertext()
                encoder.encode(slots.ravel().tolist(), SCALE, plain)
                encryptor.encrypt(plain, cipher)
                result[f"{block:06d}.bin"] = files.dump(cipher)
        return result

    def evaluate(self, encrypted, public, public_files, threads, intermediate_dir):
        if public["activation"] != "poly2" or public["sensitive_indices"] != SENSITIVE or public["feature_count"] != 8:
            raise ValueError("Expected the published eight-feature poly2 GCN")
        if set(public_files) != {"parameters.bin", "public.key", "relin.key", "galois.key"}:
            raise ValueError("Evaluator receives only public parameters and evaluation keys")
        shape = json.loads(encrypted["shape.json"])
        n = public["node_count"]
        if shape != {"nodes": n, "stride": STRIDE, "rows": ROWS}:
            raise ValueError("Ciphertext layout does not match the workload")
        start = time.perf_counter()
        arithmetic_seconds = 0.0
        with tempfile.TemporaryDirectory(prefix="scam-ckks-evaluate-") as temporary:
            files = Files(temporary)
            ctx = context(files.load(seal.EncryptionParameters, public_files["parameters.bin"]))
            engine = Arithmetic(ctx, files.load(seal.PublicKey, public_files["public.key"], ctx),
                                files.load(seal.RelinKeys, public_files["relin.key"], ctx),
                                files.load(seal.GaloisKeys, public_files["galois.key"], ctx))
            blocks = []
            for block, offset in enumerate(range(0, n, ROWS)):
                cipher = files.load(seal.Ciphertext, encrypted[f"{block:06d}.bin"], ctx)
                t = time.perf_counter()
                slots = np.zeros((ROWS, STRIDE))
                count = min(ROWS, n - offset)
                slots[:count, public["public_indices"]] = np.asarray(public["x_public"][offset:offset + count])
                engine.evaluator.add_plain_inplace(cipher, engine.plain(slots.ravel(), cipher.parms_id()))
                blocks.append(cipher)
                arithmetic_seconds += time.perf_counter() - t
            for index, name in enumerate(LAYERS):
                print(f"[toy_ckks] {name}: {n} events, {len(blocks)} ciphertext blocks", flush=True)
                t = time.perf_counter()
                blocks = engine.project(blocks, public["weights"][name + ".weight"])
                blocks = engine.aggregate(blocks, public["adjacency"])
                blocks = [engine.add_bias(v, public["weights"][name + ".bias"]) for v in blocks]
                if index < 3:
                    blocks = [engine.activation(v) for v in blocks]
                arithmetic_seconds += time.perf_counter() - t
            result = {"shape.json": encrypted["shape.json"]}
            for block, offset in enumerate(range(0, n, ROWS)):
                original = files.load(seal.Ciphertext, encrypted[f"{block:06d}.bin"], ctx)
                t = time.perf_counter()
                score = engine.score(blocks[block], original, min(ROWS, n - offset))
                arithmetic_seconds += time.perf_counter() - t
                result[f"{block:06d}.bin"] = files.dump(score)
            total = time.perf_counter() - start
            timings = {"Encrypted computation": arithmetic_seconds, "I/O": files.seconds,
                       "Context and other setup": max(0.0, total - arithmetic_seconds - files.seconds), "Total": total}
        # BERT-compatible optional flat dictionary of step name -> seconds.
        timing_path = Path(intermediate_dir) / "server_reported_steps.json"
        timing_path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
        return result

    def decrypt(self, encrypted_scores, private_files, threads):
        shape = json.loads(encrypted_scores["shape.json"])
        if shape["stride"] != STRIDE or shape["rows"] != ROWS:
            raise ValueError("Unexpected score layout")
        scores = []
        with tempfile.TemporaryDirectory(prefix="scam-ckks-decrypt-") as temporary:
            files = Files(temporary)
            ctx = context(files.load(seal.EncryptionParameters, private_files["parameters.bin"]))
            secret = files.load(seal.SecretKey, private_files["secret.key"], ctx)
            decryptor, encoder = seal.Decryptor(ctx, secret), seal.CKKSEncoder(ctx)
            for block, offset in enumerate(range(0, shape["nodes"], ROWS)):
                cipher = files.load(seal.Ciphertext, encrypted_scores[f"{block:06d}.bin"], ctx)
                plain = seal.Plaintext()
                decryptor.decrypt(cipher, plain)
                values = encoder.decode_double(plain)
                scores.extend(values[i * STRIDE] for i in range(min(ROWS, shape["nodes"] - offset)))
        return scores
