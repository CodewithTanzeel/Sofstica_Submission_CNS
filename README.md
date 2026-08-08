# DNS Exfiltration Detection — Hackathon Planning Docs

Read in this order:

1. **00_PROJECT_PLAN.md** — scope decisions, constraints, timeline, deliverables checklist
2. **01_DATA_CONTRACT.md** — fill this in first once you have the real data; both of you code against it
3. **02_TASK_SPLIT.md** — who does what (Person A: data/eval, Person B: model/explanation)
4. **03_TEST_SPECS.md** — write these test stubs together in Hour 1, then split and implement to pass them

## First 45 minutes, concretely
1. Download the organizer package, verify checksum.
2. Open the schema/data dictionary, fill in `01_DATA_CONTRACT.md` with real
   column names.
3. Scaffold the repo folders from `03_TEST_SPECS.md`.
4. Write the failing test stubs together.
5. Split: A takes `load.py`/`leakage.py`/`baseline.py`, B takes
   `features.py`/`model.py` against a mocked CSV.

Come back to me with the real schema/column names once you have them and I'll
help you fill in the data contract and generate the actual starter code
(loader, leakage guards, baseline, metrics module) against it.
