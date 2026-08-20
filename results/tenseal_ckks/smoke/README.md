# TenSEAL CKKS batch-1 smoke results

These reports were produced by the real `tenseal-ckks-node-packed` submission
on Windows 11, Python 3.12.13, and TenSEAL 0.3.16 over Microsoft SEAL. They use
the published YelpChi model checksum
`78f0bfa62d4bf88c49b82667596ed8e41ca70ac70ae77a17c1dd9d29954a4337`.

| k/L | Online latency | Peak RAM | Storage | Communication | Max score error |
|---:|---:|---:|---:|---:|---:|
| 5/22 | 10.541 s | 1,365,864,448 B | 465,403,714 B | 285,631,832 B | 2.12e-9 |
| 9/22 | 10.824 s | 1,366,941,696 B | 465,395,708 B | 285,626,156 B | 2.38e-9 |
| 14/22 | 11.073 s | 1,366,904,832 B | 465,391,121 B | 285,622,046 B | 5.22e-9 |
| 18/22 | 11.036 s | 1,366,839,296 B | 465,386,951 B | 285,620,702 B | 7.58e-9 |
| 22/22 | 10.843 s | 1,367,011,328 B | 465,405,053 B | 285,632,492 B | 3.76e-9 |

Each point used one measured run and one anomalous target plus its 98-node
one-hop context. The individual JSON files contain every stage time, artifact
size, directional byte count, key measurement, cryptographic parameter, and
quality field. `Q(k)=1` here only shows that the protected result matched the
single-target decision. It is diagnostic and does not replace the published
18,384-node plaintext quality baseline or a future full-test FHE quality run.

Communication and storage are nearly constant because this reference backend
packs each node's fixed 72-feature identifier histogram into one ciphertext,
independent of `k`.
