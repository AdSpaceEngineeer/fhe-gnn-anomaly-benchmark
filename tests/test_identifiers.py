import unittest

from fhe_gnn_anomaly_benchmark.identifiers import (
    encode_identifier,
    generate_identifiers,
    generate_iban_like,
    MAX_IDENTIFIER_LENGTH,
    retained_character_count,
    split_identifier,
    truncate_identifier,
)


def _iban_remainder(identifier: str) -> int:
    rearranged = identifier[4:] + identifier[:4]
    converted = "".join(
        str(ord(char) - 55) if char.isalpha() else char for char in rearranged
    )
    remainder = 0
    for digit in converted:
        remainder = (remainder * 10 + int(digit)) % 97
    return remainder


class IdentifierTests(unittest.TestCase):
    def test_generator_is_deterministic_and_valid(self) -> None:
        identifier = generate_iban_like("node-7", "group-2", seed=19)
        self.assertEqual(
            identifier,
            generate_iban_like("node-7", "group-2", seed=19),
        )
        self.assertEqual(len(identifier), 22)
        self.assertTrue(identifier.isalnum())
        self.assertEqual(_iban_remainder(identifier), 1)

    def test_group_controls_shared_components(self) -> None:
        first = generate_iban_like("node-1", "group-2", seed=19)
        second = generate_iban_like("node-9", "group-2", seed=19)
        self.assertEqual(first[4:18], second[4:18])
        self.assertNotEqual(first, second)

    def test_balanced_truncation(self) -> None:
        identifier = "GB12ABCD12345678901234"
        truncated = truncate_identifier(identifier, 0.2)
        self.assertEqual(truncated, identifier[:3] + identifier[-2:])

    def test_invalid_retention(self) -> None:
        for retention in (0, -0.1, 1.1):
            with self.subTest(retention=retention):
                with self.assertRaises(ValueError):
                    truncate_identifier("GB12ABCD", retention)

    def test_identifier_profile_length_limit(self) -> None:
        self.assertEqual(len(generate_iban_like(1, 1)), 22)
        with self.assertRaises(ValueError):
            truncate_identifier("A" * (MAX_IDENTIFIER_LENGTH + 1), 0.5)

    def test_realized_retention_and_encoding(self) -> None:
        identifier = "GB12ABCD12345678901234"
        self.assertEqual(retained_character_count(len(identifier), 0.2), 5)
        prefix, suffix = split_identifier(identifier, 5)
        self.assertEqual((prefix, suffix), (identifier[:3], identifier[-2:]))
        encoded = encode_identifier(identifier, retention=0.2)
        self.assertEqual(encoded.retained_characters, 5)
        self.assertEqual(encoded.sides, (0, 0, 0, 1, 1))
        self.assertEqual(encoded.retention, 5 / 22)

    def test_batch_generator_rejects_misalignment(self) -> None:
        with self.assertRaises(ValueError):
            generate_identifiers([0, 1], ["group"], seed=3)


if __name__ == "__main__":
    unittest.main()
