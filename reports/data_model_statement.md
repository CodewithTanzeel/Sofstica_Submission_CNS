# Data and Model Statement

Required deliverable #5 per the challenge brief.

## Dataset

- **Name / source:** CIC-Bell-DNS-EXF-2021, Canadian Institute for
  Cybersecurity, University of New Brunswick, in collaboration with Bell
  Canada Cyber Threat Intelligence.
  https://www.unb.ca/cic/datasets/dns-exf-2021.html
- **Required citation:** Samaneh Mahdavifar, Amgad Hanafy Salem, Princy
  Victor, Miguel Garzon, Amir H. Razavi, Natasha Hellberg, Arash Habibi
  Lashkari, "Lightweight Hybrid Detection of Data Exfiltration using DNS
  based on Machine Learning," 11th IEEE International Conference on
  Communication and Network Security (ICCNS), Dec. 3-5, 2021, Beijing
  Jiaotong University, Weihai, China.
- **Organizer snapshot version / SHA-256:** `>>> FILL IN — copy the exact
  snapshot version string and archive checksum from the organizer download
  page before submitting. This must match the source-of-truth checksum
  published by the organizers, not a value you compute yourself unless the
  page directs you to. <<<`
- **Files actually used:** all stateful/stateless CSVs under `data/Benign/`,
  `data/Attack_Light_Benign/`, `data/Attack_heavy_Benign/` (list is in
  `results/group_split_manifest.csv`).

## External data / pretrained models / APIs

None. No external datasets, no pretrained/foundation models, no third-party
ML APIs were used. The model (XGBoost + a logistic regression baseline) is
trained from scratch on the organizer-provided snapshot only.

## Third-party services / data transfer

None. The prototype runs entirely locally; no PCAPs, feature rows, or
derived records are uploaded to any external service. This satisfies the
brief's safety rule #4 ("Do not upload PCAPs, domain data, or derived
records to an external service unless event rules and the data licence
explicitly allow it").

## Software / library versions

See `requirements.txt`. Exact versions used for the reported score should be
filled in via `pip freeze` immediately before final submission so the
lockfile matches the run that produced `results/scoring_output.json`.

## Licence compliance

Use and redistribution of CIC-Bell-DNS-EXF-2021 follows the licence and
citation terms on the official UNB page (linked above). No modified or
re-hosted copies of the raw dataset are included in this submission beyond
what the organizer package already provides.
