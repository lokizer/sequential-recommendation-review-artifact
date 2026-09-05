# Data documentation

## Status

Beauty, LastFM, and Toys and Games are **reference data**. They were not
collected by the DuoSpectra authors. The processed files are inherited from the
public BSARec research repository and are included only to reproduce the exact
experimental protocol.

| File | Users | SHA-256 |
| --- | ---: | --- |
| `src/data/Beauty.txt` | 22,363 | `f85a84ab47d70e82bf0e7301804270af10fb51e127ab8f0885697d593f6f6d43` |
| `src/data/LastFM.txt` | 1,090 | `46767eacbf73bce98cc6c402671933a22ede8d73d3433c0e164d6b1868775a62` |
| `src/data/Toys_and_Games.txt` | 19,412 | `fb28d2f6bc01d38e0da8ad88265d62043364f1a908e8dbf3591d5bba68c9d789` |

Each line has the format `user_id item_id_1 item_id_2 ... item_id_n`. IDs are
integer indices. Interactions are ordered chronologically. The last item is the
test target and the penultimate item is the validation target. The model keeps
at most the 50 most recent input items and left-pads shorter sequences.

## Sources

- Beauty and Toys and Games originate from the Amazon Review Data published by
  Julian McAuley and collaborators:
  <https://snap.stanford.edu/data/amazon/productGraph/>
- LastFM originates from the HetRec 2011 Last.fm 2K dataset:
  <https://lenskit.grouplens.org/datasets/hetrec-2011/>
- The preprocessing follows the public S3-Rec data-processing code and the
  BSARec repository. Reprocessing utilities are in `src/data/process/`.

## Required citations

Users of this artifact should cite the relevant dataset sources in addition to
the DuoSpectra and BSARec papers:

- J. McAuley, C. Targett, Q. Shi, and A. van den Hengel. Image-based
  Recommendations on Styles and Substitutes. SIGIR, 2015.
- I. Cantador, P. Brusilovsky, and T. Kuflik. Second Workshop on Information
  Heterogeneity and Fusion in Recommender Systems (HetRec 2011). RecSys, 2011.

Review the upstream dataset README and terms before reuse or redistribution.
No ownership of the source datasets is claimed by this artifact.
