# Running Experiment 1: FHP SD-CFR efficiency audit

For **Experiment 4 (Experiment 3 on n2-standard-16)**, see its
[configuration and launch guide](../experiments/fhp/exp4_sd_cfr_structured_n2_standard16/README.md).

For **Experiment 3 (structured player inputs)**, see its
[configuration and launch guide](../experiments/fhp/exp3_sd_cfr_structured_24h/README.md).

For the new **three-seed 24-hour Experiment 2**, see the
[dedicated configuration and cloud launch guide](../experiments/fhp/exp2_sd_cfr_24h/README.md).
It uses a remote controller and three separate training VMs; the instructions
below remain specifically for the short Experiment 1 implementation audit.

The active Experiment 1 compares the reference SD-CFR implementation with faster
inference and lossless replay compression. It is not the former conventional
Deep CFR training run, now retained under `archive_exp1_deep_cfr_best_config_transfer`.
See the [full experiment specification](../experiments/fhp/exp1_sd_cfr_efficiency/README.md).

## One-time setup

The [archived cloud guide](archive_GCP_DEEP_CFR_EXPERIMENT1.md) retains the detailed
project, API, bucket and service-account setup in sections 1--7. Those shared
setup steps still apply, but its old training submission commands do not launch
the current experiment. Set `PROJECT_ID`, `REGION`, `BUCKET` (including `gs://`)
and `SA_EMAIL` in the terminal. Commit and push the experiment before submission.

## Launch

From the FHP Deep CFR repository:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="fhp-sdcfr-exp1-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp1_sd_cfr_efficiency.sh run
```

One `n2-standard-8` VM runs a smoke test, then the full benchmark only if the
smoke passes. Three seeds, three timing repeats and three arms run sequentially
to avoid competing for CPU. The job cap is two hours including setup. No child
job controller is used. The launcher rejects refs missing the experiment code.

## Monitor and download

```bash
gcloud batch jobs describe "$RUN_ID" --project "$PROJECT_ID" --location "$REGION"
gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION"

mkdir -p "outputs/downloaded/$RUN_ID"
gcloud storage rsync --recursive "$BUCKET/$RUN_ID/outputs/$RUN_ID" "outputs/downloaded/$RUN_ID"
```

Retain the run ID. Reports, validation checks, timings and charts are uploaded;
temporary replay and optimiser comparison payloads are not retained. No model
or training configuration is promoted automatically from a timing result.
