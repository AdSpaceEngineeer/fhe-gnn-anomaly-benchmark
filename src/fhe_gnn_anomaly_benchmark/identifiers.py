"""Deterministic synthetic IBAN-form identifiers.

The generated values are synthetic test data. They are structurally similar to
GB IBANs and satisfy the ISO 13616 mod-97 check, but they do not identify real
accounts.
"""

from __future__ import annotations

import hashlib
import math
import string


def _digest(seed: int, namespace: str, value: str) -> bytes:
    payload = f"{seed}:{namespace}:{value}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _letters(data: bytes, length: int) -> str:
    return "".join(string.ascii_uppercase[byte % 26] for byte in data[:length])


def _digits(data: bytes, length: int) -> str:
    integer = int.from_bytes(data, "big")
    return str(integer % (10**length)).zfill(length)


def _iban_numeric(value: str) -> str:
    converted: list[str] = []
    for char in value:
        converted.append(str(ord(char) - 55) if char.isalpha() else char)
    return "".join(converted)


def _check_digits(country: str, bban: str) -> str:
    rearranged = f"{bban}{country}00"
    remainder = 0
    for digit in _iban_numeric(rearranged):
        remainder = (remainder * 10 + int(digit)) % 97
    return str(98 - remainder).zfill(2)


def generate_iban_like(node_id: str | int, group_id: str | int, seed: int = 0) -> str:
    """Return a deterministic 22-character synthetic GB-format IBAN.

    ``group_id`` controls the four-letter bank code and the first four account
    digits. Nodes in the same declared relational group therefore share both a
    prefix and an account-number component without consulting anomaly labels.
    """

    country = "GB"
    group_hash = _digest(seed, "group", str(group_id))
    node_hash = _digest(seed, "node", str(node_id))
    bank_code = _letters(group_hash, 4)
    sort_code = _digits(group_hash[4:], 6)
    account_number = _digits(group_hash[10:], 4) + _digits(node_hash, 4)
    bban = f"{bank_code}{sort_code}{account_number}"
    return f"{country}{_check_digits(country, bban)}{bban}"


def truncate_identifier(identifier: str, retention: float) -> str:
    """Retain a balanced prefix and suffix at the requested fraction.

    At least two characters are retained so both sides remain represented.
    The full identifier is returned when rounding reaches its length.
    """

    if not identifier:
        raise ValueError("identifier must not be empty")
    if not 0 < retention <= 1:
        raise ValueError("retention must be in the interval (0, 1]")

    retained = min(len(identifier), max(2, math.ceil(len(identifier) * retention)))
    prefix_length = math.ceil(retained / 2)
    suffix_length = retained // 2
    if retained == len(identifier):
        return identifier
    return identifier[:prefix_length] + identifier[-suffix_length:]
