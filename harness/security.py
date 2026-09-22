"""Evidence gate; this is not a universal cryptographic security proof."""
import math
from harness.params import MIN_SECURITY_BITS, SEAL_TC128_MAX_BITS


def validate_description(description, debug=False):
    if description.get("is_fhe") is False:
        if not debug:
            raise ValueError("Plaintext adapter requires --debug-plaintext; never eligible as FHE")
        return {"status": "not_fhe", "eligible": False}
    if description.get("is_fhe") is not True:
        raise ValueError("Submission must declare is_fhe")
    security = description.get("security", {})
    bits = security.get("classical_bits", 0)
    if not isinstance(bits, (int, float)) or not math.isfinite(bits) or bits < MIN_SECURITY_BITS:
        raise ValueError("At least 128-bit classical security is required")
    if not security.get("evidence") or not description.get("parameters"):
        raise ValueError("Provide cryptographic parameters and security evidence")
    for name in ("encoding", "packing", "activation"):
        if not description.get(name):
            raise ValueError("Missing submission description: " + name)
    if security.get("validator") == "seal_tc128":
        p = description["parameters"]
        n, chain = p.get("poly_modulus_degree"), p.get("coeff_mod_bit_sizes")
        if n not in SEAL_TC128_MAX_BITS or not isinstance(chain, list) or not chain or any(type(b) is not int or not 2 <= b <= 60 for b in chain):
            raise ValueError("Invalid SEAL parameters")
        if sum(chain) > SEAL_TC128_MAX_BITS[n]:
            raise ValueError("Coefficient modulus exceeds SEAL's 128-bit security bound")
        return {"status": "awaiting_context_check", "eligible": False, "classical_bits": 128}
    return {"status": "evidence_requires_review", "eligible": False, "claimed_classical_bits": bits}


def inspect_public_context(description, public_keys, threads):
    """Check the actual serialized TenSEAL context, not just claimed parameters."""
    if description.get("security", {}).get("validator") != "seal_tc128":
        return None
    import tenseal as ts
    context = ts.context_from(public_keys["context.bin"], n_threads=threads)
    if context.has_secret_key():
        raise ValueError("Server context contains a secret key")
    seal = context.seal_context().data
    key = seal.key_context_data()
    parms = key.parms()
    actual_degree = parms.poly_modulus_degree()
    declared = description["parameters"]
    # TenSEAL 0.3.16 does not bind seal::Modulus on every platform. Compare
    # parameter fingerprints to a context built from the declared chain instead.
    expected = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=declared["poly_modulus_degree"],
                          coeff_mod_bit_sizes=declared["coeff_mod_bit_sizes"], n_threads=threads)
    expected_id = expected.seal_context().data.key_context_data().parms_id()
    if actual_degree != declared["poly_modulus_degree"] or key.parms_id() != expected_id:
        raise ValueError("Actual encryption context differs from submission parameters")
    if not key.qualifiers().parameters_set() or key.total_coeff_modulus_bit_count() > SEAL_TC128_MAX_BITS[actual_degree]:
        raise ValueError("Serialized encryption context failed security checks")
    return {"status": "seal_tc128_context_checked", "eligible": True, "classical_bits": 128,
            "actual_poly_modulus_degree": actual_degree, "actual_coeff_mod_bit_sizes": declared["coeff_mod_bit_sizes"],
            "scope": "SEAL default tc128 parameter validation; implementation and threat assumptions still require review"}
