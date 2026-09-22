"""Simple packed SEAL operations. No model training, shortcuts or decryption.

Each ciphertext holds 256 event rows, with 64 slots reserved per event.
Public sparse graph coefficients move complete rows, including across ciphertexts.
The implementation favors explicit equations over optimized rotation schedules.
"""
from pathlib import Path
import time
import numpy as np
import tenseal.sealapi as seal

DEGREE = 32768
CHAIN = [60] + [40] * 16 + [60]  # 760 bits < SEAL tc128 bound 881
SCALE = float(2**40)
STRIDE = 64
SLOTS = DEGREE // 2
ROWS = SLOTS // STRIDE
SENSITIVE = [4, 6, 7]
LAYERS = ("encoder_1", "encoder_2", "decoder_1", "decoder_2")


def parameters():
    p = seal.EncryptionParameters(seal.SCHEME_TYPE.CKKS)
    p.set_poly_modulus_degree(DEGREE)
    p.set_coeff_modulus(seal.CoeffModulus.Create(DEGREE, CHAIN))
    return p


def context(p):
    c = seal.SEALContext(p, True, seal.SEC_LEVEL_TYPE.TC128)
    if not c.parameters_set():
        raise ValueError("SEAL rejected the 128-bit security parameters")
    return c


class Files:
    """SEAL's Python save/load API uses paths; account for this file I/O."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.seconds = 0.0

    def dump(self, obj):
        start = time.perf_counter()
        path = self.directory / "serialize.tmp"
        try:
            obj.save(str(path))
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
            self.seconds += time.perf_counter() - start

    def load(self, kind, data, ctx=None):
        start = time.perf_counter()
        path = self.directory / "deserialize.tmp"
        try:
            path.write_bytes(data)
            obj = kind(seal.SCHEME_TYPE.CKKS) if kind is seal.EncryptionParameters else kind()
            if ctx is None:
                obj.load(str(path))
            else:
                obj.load(ctx, str(path))
            return obj
        finally:
            path.unlink(missing_ok=True)
            self.seconds += time.perf_counter() - start


def feature_diagonals(weights):
    """For y[row,out] = sum_in x[row,in] W[in,out], without row wrap."""
    weights = np.asarray(weights, dtype=float)
    ni, no = weights.shape
    if max(ni, no) > STRIDE:
        raise ValueError("Published model widths must fit the fixed 64-slot row layout")
    for shift in range(1 - no, ni):
        mask = np.zeros(STRIDE)
        out = np.arange(max(0, -shift), min(no, ni - shift))
        mask[out] = weights[out + shift, out]
        if np.any(mask):
            yield shift, np.tile(mask, ROWS)


def graph_terms(graph):
    """Group COO edges by source ciphertext, row rotation and destination block."""
    row, col = np.asarray(graph["rows"], dtype=np.int64), np.asarray(graph["cols"], dtype=np.int64)
    values = np.asarray(graph["values"], dtype=float)
    src, dst = col // ROWS, row // ROWS
    shift = (col % ROWS - row % ROWS) % ROWS
    order = np.lexsort((dst, shift, src))
    key = np.column_stack((src[order], shift[order], dst[order]))
    if not len(order):
        return
    cuts = np.r_[0, 1 + np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)), len(order)]
    for left, right in zip(cuts[:-1], cuts[1:]):
        selected = order[left:right]
        mask = np.zeros(ROWS)
        np.add.at(mask, row[selected] % ROWS, values[selected])
        if np.any(mask):
            yield int(src[selected[0]]), int(shift[selected[0]]) * STRIDE, int(dst[selected[0]]), np.repeat(mask, STRIDE)


class Arithmetic:
    def __init__(self, ctx, public_key, relin, galois):
        self.ctx = ctx
        self.encoder = seal.CKKSEncoder(ctx)
        self.evaluator = seal.Evaluator(ctx)
        self.encryptor = seal.Encryptor(ctx, public_key)
        self.relin, self.galois = relin, galois

    def plain(self, values, pid, scale=SCALE):
        p = seal.Plaintext()
        self.encoder.encode(values.tolist() if isinstance(values, np.ndarray) else values, pid, scale, p)
        return p

    def rotate(self, value, shift):
        # Generate only positive power-of-two rotation keys; compose arbitrary shifts.
        shift %= SLOTS
        output = value
        bit = 1
        while shift:
            if shift & 1:
                rotated = seal.Ciphertext()
                self.evaluator.rotate_vector(output, bit, self.galois, rotated)
                output = rotated
            bit *= 2
            shift >>= 1
        return output

    def product(self, value, plain):
        result = seal.Ciphertext()
        self.evaluator.multiply_plain(value, plain, result)
        return result

    def accumulate(self, total, value):
        if total is None:
            return value
        self.evaluator.add_inplace(total, value)
        return total

    def rescale(self, value):
        self.evaluator.rescale_to_next_inplace(value)
        value.scale = SCALE
        return value

    def switch(self, value, pid):
        out = seal.Ciphertext()
        self.evaluator.mod_switch_to(value, pid, out)
        out.scale = SCALE
        return out

    def zero_next(self, value):
        out = seal.Ciphertext()
        pid = self.ctx.get_context_data(value.parms_id()).next_context_data().parms_id()
        self.encryptor.encrypt_zero(pid, out)
        out.scale = SCALE
        return out

    def project(self, blocks, weights):
        masks = [(shift, self.plain(mask, blocks[0].parms_id())) for shift, mask in feature_diagonals(weights)]
        result = []
        for block in blocks:
            total = None
            for shift, mask in masks:
                total = self.accumulate(total, self.product(self.rotate(block, shift), mask))
            result.append(self.rescale(total) if total is not None else self.zero_next(block))
        return result

    def aggregate(self, blocks, graph):
        result = [None] * len(blocks)
        previous, rotated = None, None
        groups = 0
        for source, shift, destination, mask in graph_terms(graph):
            if previous != (source, shift):
                rotated = self.rotate(blocks[source], shift)
                previous = source, shift
            term = self.product(rotated, self.plain(mask, rotated.parms_id()))
            result[destination] = self.accumulate(result[destination], term)
            groups += 1
            if groups % 10000 == 0:
                print(f"[toy_ckks] processed {groups} graph block terms", flush=True)
        return [self.rescale(v) if v is not None else self.zero_next(blocks[i]) for i, v in enumerate(result)]

    def add_bias(self, value, bias):
        mask = np.zeros(STRIDE)
        mask[:len(bias)] = bias
        if np.any(mask):
            self.evaluator.add_plain_inplace(value, self.plain(np.tile(mask, ROWS), value.parms_id()))
        return value

    def square(self, value):
        result = seal.Ciphertext()
        self.evaluator.square(value, result)
        self.evaluator.relinearize_inplace(result, self.relin)
        return self.rescale(result)

    def activation(self, value):
        square = self.square(value)
        scaled = self.rescale(self.product(square, self.plain(0.125, square.parms_id())))
        self.evaluator.add_inplace(scaled, self.switch(value, scaled.parms_id()))
        return scaled

    def score(self, reconstructed, original, count):
        error = seal.Ciphertext()
        self.evaluator.sub(reconstructed, self.switch(original, reconstructed.parms_id()), error)
        squared = self.square(error)
        mask = np.zeros(SLOTS)
        mask[np.arange(count) * STRIDE] = 1.0 / len(SENSITIVE)
        plain = self.plain(mask, squared.parms_id())
        total = None
        for feature in SENSITIVE:
            total = self.accumulate(total, self.product(self.rotate(squared, feature), plain))
        # Other slots are masked: the result contains scores, not feature-wise errors.
        return self.rescale(total)
