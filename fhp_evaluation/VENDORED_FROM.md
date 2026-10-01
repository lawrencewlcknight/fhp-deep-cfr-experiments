# Shared FHP evaluation snapshot

FHP SD-CFR Experiment 2 copies this snapshot from the UCV-ESCHER repository.
Evaluator changes are an optional `begin_episode` hook in duplicate play for
historical-mixture policies and an opt-in split seed layout matching the
completed UCV retrospective analysis. Default chance/action streams and
stateless-policy results are unchanged. SD-CFR LBR queries use a separate
exact own-reach behavioural-mixture adapter; the sampled hidden network is
never supplied to the responder. The upstream snapshot used by the completed
UCV evaluation has source-tree SHA256
`c61209654661d1ad8dd4e716f68aa56a170655b7d1fe3f85545db1850e2e1a79`.

This package is a byte-for-byte source snapshot of the locally validated
`fhp-evaluation-suite/fhp_evaluation` package as of 2026-09-22. It is vendored
so that cloud evaluation jobs can reconstruct the evaluator from the pinned
`fhp-ucv-escher-experiments` commit without depending on files from the
submitting laptop.

The original suite was validated jointly against UCV-ESCHER, VR-Deep and Deep
CFR checkpoints. Its provenance includes the five rule agents and pre-flop
lookup from the released DeepPDCFR artefact; the LooseAggressive thresholds are
corrected to the intended increasing order `(-300, -100)`. Local Best Response
is reported only as a lower bound on best-response value, never as exact
exploitability.
