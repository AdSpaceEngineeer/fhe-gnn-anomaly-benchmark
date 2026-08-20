import unittest

from fhe_gnn_anomaly_benchmark.identifiers import (
    generate_iban_like,
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


if __name__ == "__main__":
    unittest.main()
