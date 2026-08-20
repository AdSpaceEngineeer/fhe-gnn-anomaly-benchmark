"""Deterministic synthetic IBAN-form identifiers.

The generated values are synthetic test data. They are structurally similar to
GB IBANs and satisfy the ISO 13616 mod-97 check, but they do not identify real
accounts.
"""

from __future__ import annotations

import hashlib
import math
import string
from dataclasses import dataclass
from typing import Iterable, Sequence


IDENTIFIER_ALPHABET = string.digits + string.ascii_uppercase
ALPHABET_INDEX = {character: index for index, character in enumerate(IDENTIFIER_ALPHABET)}
MIN_IDENTIFIER_LENGTH = 2
MAX_IDENTIFIER_LENGTH = 64
CANONICAL_RETENTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True, slots=True)
class EncodedIdentifier:
    """A balanced prefix/suffix identifier representation.

    ``values`` contains alphabet indices and ``sides`` marks prefix values with
    0 and suffix values with 1.  FHE submissions encrypt these values; the
    harness keeps the exact representation scheme-independent.
    """

    values: tuple[int, ...]
    sides: tuple[int, ...]
    retained_characters: int
    full_length: int

    @property
    def retention(self) -> float:
        return self.retained_characters / self.full_length


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


def generate_identifiers(
    node_ids: Iterable[str | int],
    group_ids: Iterable[str | int],
    *,
    seed: int = 0,
) -> list[str]:
    """Generate one identifier per node from predeclared relational groups."""

    nodes = list(node_ids)
    groups = list(group_ids)
    if len(nodes) != len(groups):
        raise ValueError("node_ids and group_ids must have the same length")
    return [
        generate_iban_like(node_id, group_id, seed=seed)
        for node_id, group_id in zip(nodes, groups, strict=True)
    ]


def retained_character_count(full_length: int, retention: float) -> int:
    """Return the realized integer ``k`` for a requested retention fraction."""

    validate_identifier_length(full_length)
    if not 0 < retention <= 1:
        raise ValueError("retention must be in the interval (0, 1]")
    return min(full_length, max(MIN_IDENTIFIER_LENGTH, math.ceil(full_length * retention)))


def validate_identifier_length(length: int) -> None:
    """Validate the cross-profile identifier bound used by the harness."""

    if not MIN_IDENTIFIER_LENGTH <= length <= MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"identifier length must be between {MIN_IDENTIFIER_LENGTH} "
            f"and {MAX_IDENTIFIER_LENGTH} characters"
        )


def split_identifier(identifier: str, retained_characters: int) -> tuple[str, str]:
    """Return balanced prefix and suffix strings totalling exactly ``k``."""

    validate_identifier_length(len(identifier))
    if not 2 <= retained_characters <= len(identifier):
        raise ValueError("retained_characters must be between 2 and identifier length")
    prefix_length = math.ceil(retained_characters / 2)
    suffix_length = retained_characters // 2
    return identifier[:prefix_length], identifier[-suffix_length:]


def encode_identifier(
    identifier: str,
    *,
    retention: float | None = None,
    retained_characters: int | None = None,
) -> EncodedIdentifier:
    """Encode a retained prefix/suffix as base-36 character indices.

    Exactly one of ``retention`` or ``retained_characters`` must be supplied.
    Values are deliberately not one-hot encoded by the client: submissions may
    choose their own FHE-efficient encrypted lookup or equality strategy while
    preserving the same logical input.
    """

    if (retention is None) == (retained_characters is None):
        raise ValueError("provide exactly one of retention or retained_characters")
    normalized = identifier.upper()
    validate_identifier_length(len(normalized))
    invalid = sorted(set(normalized) - set(IDENTIFIER_ALPHABET))
    if invalid:
        raise ValueError(f"identifier contains unsupported characters: {invalid}")
    if retained_characters is None:
        retained_characters = retained_character_count(len(normalized), retention)  # type: ignore[arg-type]
    prefix, suffix = split_identifier(normalized, retained_characters)
    characters = prefix + suffix
    return EncodedIdentifier(
        values=tuple(ALPHABET_INDEX[character] for character in characters),
        sides=(0,) * len(prefix) + (1,) * len(suffix),
        retained_characters=retained_characters,
        full_length=len(normalized),
    )


def encode_identifiers(
    identifiers: Sequence[str],
    *,
    retention: float,
) -> list[EncodedIdentifier]:
    """Encode a batch and reject mixed identifier lengths."""

    if not identifiers:
        raise ValueError("identifiers must not be empty")
    lengths = {len(identifier) for identifier in identifiers}
    if len(lengths) != 1:
        raise ValueError("all identifiers in a benchmark batch must have equal length")
    return [encode_identifier(identifier, retention=retention) for identifier in identifiers]


def truncate_identifier(identifier: str, retention: float) -> str:
    """Retain a balanced prefix and suffix at the requested fraction.

    At least two characters are retained so both sides remain represented.
    The full identifier is returned when rounding reaches its length.
    """

    retained = retained_character_count(len(identifier), retention)
    if retained == len(identifier):
        return identifier
    prefix, suffix = split_identifier(identifier, retained)
    return prefix + suffix
