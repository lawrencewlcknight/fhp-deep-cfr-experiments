# Running FHP Deep CFR experiments on Google Cloud Batch

This guide explains how to run this repository's experiments on Google Cloud
Batch from a local terminal. The workflow is:

1. prepare a Google Cloud project;
2. create a Cloud Storage bucket for durable outputs;
3. create a least-privilege service account for Batch VMs;
4. verify and publish the repository revision to run;
5. submit a smoke test;
6. monitor the job and inspect its logs;
7. submit the full Experiment 1 run;
8. download and verify the outputs.

The repository already contains `gcp/submit_batch_experiment.sh`; do not create
a second submission script. The helper generates a Batch job definition, starts
a temporary VM, clones this repository, creates an isolated virtual environment,
installs CPU-only dependencies, runs the requested module, uploads `outputs/`
to Cloud Storage every 30 minutes and once more on exit, and prints monitoring
commands. Batch manages the VM lifecycle.

The commands below follow the current Google Cloud documentation for
[Batch IAM and job creation](https://docs.cloud.google.com/batch/docs/get-started),
[custom service accounts](https://docs.cloud.google.com/batch/docs/create-run-job-custom-service-account),
[Cloud Logging](https://docs.cloud.google.com/batch/docs/analyze-job-using-logs),
[custom boot disks](https://docs.cloud.google.com/batch/docs/create-run-job-custom-boot-disks),
and [uniform bucket-level access](https://docs.cloud.google.com/storage/docs/uniform-bucket-level-access).

---

## 1. Prerequisites

You need:

- a Google Cloud project with billing enabled;
- the Google Cloud CLI (`gcloud`) installed locally;
- permission to enable APIs and manage the required IAM bindings;
- permission to create Batch jobs and Cloud Storage buckets;
- the latest experiment commit pushed to the remote repository;
- a repository that the Batch VM can clone non-interactively.

The default clone URL is:

```text
https://github.com/lawrencewlcknight/fhp-deep-cfr-experiments.git
```

The current wrapper assumes that URL is publicly cloneable. For a private
repository, use an authenticated clone mechanism or a pre-built container image
before submitting a paid job.

---

## 2. Verify and publish the repository revision

From the repository root:

```bash
git status --short
git remote -v
git log -1 --oneline
git push origin main
```

`git status --short` should print nothing. Confirm that `origin` points to the
FHP Deep CFR repository. The wrapper's shallow clone accepts a branch or tag,
not a raw commit hash. For routine smoke testing, use `main`:

```bash
export REPO_REF="main"
echo "$REPO_REF"
```

For an immutable full run, create and push an annotated tag:

```bash
export EXPERIMENT_TAG="fhp-deep-cfr-exp1-v1"
git tag -a "$EXPERIMENT_TAG" -m "FHP Deep CFR Experiment 1"
git push origin "$EXPERIMENT_TAG"
export REPO_REF="$EXPERIMENT_TAG"
```

The submission helper passes `REPO_REF` to `git clone --branch`. If no value is
supplied, it uses `main`. The VM logs the resolved commit for verification.

To use a fork or differently named remote repository:

```bash
export REPO_URL="https://github.com/YOUR_ACCOUNT/fhp-deep-cfr-experiments.git"
```

---

## 3. Authenticate and select the Google Cloud project

This is normally a one-time local setup:

```bash
gcloud init
gcloud auth login

export PROJECT_ID="your-gcp-project-id"
gcloud config set project "$PROJECT_ID"
```

Choose a Batch region. For UK-based work, `europe-west1` is a reasonable
starting point, subject to machine availability and your organisation's policy:

```bash
export REGION="europe-west1"
```

Enable the APIs used by the wrapper:

```bash
gcloud services enable \
  batch.googleapis.com \
  compute.googleapis.com \
  logging.googleapis.com \
  storage.googleapis.com
```

Confirm the active identity and project before creating resources:

```bash
gcloud auth list
gcloud config get-value project
```

---

## 4. Create a Cloud Storage bucket

Bucket names are globally unique. This convention usually works because a
Google Cloud project ID is itself globally unique:

```bash
export BUCKET_NAME="${PROJECT_ID}-fhp-deep-cfr-results"
export BUCKET="gs://${BUCKET_NAME}"

gcloud storage buckets create "$BUCKET" \
  --location="$REGION" \
  --uniform-bucket-level-access
```

Verify it:

```bash
gcloud storage buckets describe "$BUCKET"
```

If the bucket already exists and belongs to this project, skip creation and
only set `BUCKET`. Uniform bucket-level access keeps object permissions under
IAM instead of per-object ACLs.

---

## 5. Create the Batch VM service account

Create a dedicated service account rather than giving experiments the default
Compute Engine identity:

```bash
export SA_NAME="fhp-deep-cfr-runner"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --display-name="FHP Deep CFR Batch runner" \
  --project="$PROJECT_ID"
```

Grant the VM permission to report Batch state and write Cloud Logging entries:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/batch.agentReporter"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"
```

Grant it object access only on the experiment bucket:

```bash
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"
```

The uploader overwrites objects during periodic uploads, so `objectAdmin` is
used instead of the create-only role.

---

## 6. Grant the submitting user the required roles

Determine the active account or set it explicitly:

```bash
export YOUR_EMAIL="your-email@example.com"
```

The submitting user needs Batch Job Editor on the project and Service Account
User on the job's service account:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/batch.jobsEditor"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/iam.serviceAccountUser"
```

To inspect task logs, also grant Logs Viewer if your existing project role does
not already contain it:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/logging.viewer"
```

Project owners may already have these permissions. Prefer the narrow roles
above for routine use.

---

## 7. Environment variables for each new terminal

Set these before submitting a job:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
export BUCKET="gs://${PROJECT_ID}-fhp-deep-cfr-results"
export SA_EMAIL="fhp-deep-cfr-runner@${PROJECT_ID}.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/fhp-deep-cfr-experiments.git"
export REPO_REF="main"

gcloud config set project "$PROJECT_ID"
```

Check all values:

```bash
printf 'PROJECT_ID=%s\nREGION=%s\nBUCKET=%s\nSA_EMAIL=%s\nREPO_URL=%s\nREPO_REF=%s\n' \
  "$PROJECT_ID" "$REGION" "$BUCKET" "$SA_EMAIL" "$REPO_URL" "$REPO_REF"
```

To make a run exactly reproducible, use the pushed tag described in section 2
rather than allowing `main` to move.

---

## 8. Verify the submission helper locally

From the repository root:

```bash
bash -n gcp/submit_batch_experiment.sh
python -m experiments.fhp.exp1_deep_cfr_best_config_transfer.run --help
pytest -q
```

`bash -n` should return silently. The wrapper also refuses to submit if the
Python module in the experiment command does not exist in this repository.

Its positional arguments are:

```text
JOB_NAME
PYTHON_EXPERIMENT_COMMAND
MACHINE_TYPE
MAX_RUN_SECONDS
CPU_MILLI
MEMORY_MIB
BOOT_DISK_SIZE_GB
BOOT_DISK_TYPE
```

The final six have defaults, but full commands should state them explicitly so
the resource request is recorded in shell history and Batch logs.

---

## 9. Run a local smoke test first

Before allocating a VM, exercise the exact Experiment 1 runner locally:

```bash
python -m experiments.fhp.exp1_deep_cfr_best_config_transfer.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 2 \
  --evaluation-interval 1 \
  --checkpoint-schedule 1,2 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-train-every 1 \
  --output-root outputs/local-smoke
```

Check that the run produced `run_manifest.json`, two policy snapshots, one full
checkpoint, and the CSV/JSON/NPZ summaries.

---

## 10. Submit a Batch smoke test

Use a unique, lowercase job name:

```bash
export JOB_NAME="fhp-deep-cfr-exp1-smoke-$(date +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_deep_cfr_best_config_transfer.run \
    --seeds 1234 \
    --iterations 2 \
    --traversals 2 \
    --evaluation-interval 1 \
    --checkpoint-schedule 1,2 \
    --policy-network-layers 8,8 \
    --advantage-network-layers 8,8 \
    --batch-size-advantage 2 \
    --batch-size-strategy 2 \
    --memory-capacity 256 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --policy-network-train-every 1 \
    --output-root outputs/cloud/$JOB_NAME" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000" \
  "100" \
  "pd-balanced"
```

Before submission, the helper prints the complete script that the VM will run.
Confirm the repository URL, revision, module, output directory, bucket, machine
type, memory, runtime, and boot disk. The smoke job validates cloning, package
installation, OpenSpiel's Universal Poker support, checkpointing, and uploads.

---

## 11. Monitor the job

List jobs in the region:

```bash
gcloud batch jobs list --location "$REGION"
```

Describe the current job:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
```

Useful states include `QUEUED`, `SCHEDULED`, `RUNNING`, `SUCCEEDED`, and
`FAILED`. Once the job has run, `status.runDuration` reports its running time.

The wrapper uploads under this layout:

```text
gs://BUCKET/JOB_NAME/outputs/...
```

List objects during or after the run:

```bash
gcloud storage ls --recursive "$BUCKET/$JOB_NAME/"
```

Periodic uploads default to every 1,800 seconds. Override the interval before
submission if needed:

```bash
export UPLOAD_INTERVAL_SECONDS=900
```

The wrapper always attempts a final upload when the task exits, including after
an experiment failure. VM termination outside the shell's control can still
prevent that final handler from running, which is why periodic uploads matter.

---

## 12. Read task logs

Get the job UID:

```bash
export JOB_UID="$(
  gcloud batch jobs describe "$JOB_NAME" \
    --location "$REGION" \
    --format='value(uid)'
)"
echo "$JOB_UID"
```

Read standard output and standard error from the Batch task:

```bash
gcloud logging read \
  "logName=\"projects/${PROJECT_ID}/logs/batch_task_logs\" AND labels.job_uid=\"${JOB_UID}\"" \
  --limit=1000 \
  --order=asc \
  --format='value(timestamp,severity,textPayload,jsonPayload.message)'
```

Look for:

- `Resolved repository commit` to confirm the code revision;
- `Starting seed` and snapshot messages for progress;
- `Outputs:` for the VM-local run directory;
- `Uploading` for Cloud Storage progress;
- `No space left on device` for disk pressure;
- `Killed`, exit code `137`, or allocation errors for memory pressure;
- `maxRunDuration` for an expired time limit;
- clone or authentication errors for repository access problems.

If task logs never appear, inspect `statusEvents` in `gcloud batch jobs
describe`; the VM may have failed before the runnable started.

---

## 13. Download and verify smoke outputs

Download the job prefix into an ignored local scratch directory:

```bash
mkdir -p "cloud_outputs/$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

Inspect the run manifest:

```bash
find "cloud_outputs/$JOB_NAME" -name run_manifest.json -print
```

The smoke run should contain:

- `run_manifest.json` with status `completed`;
- `experiment_metadata.json`;
- `policy_snapshot_manifest.csv`;
- two files ending in `_snapshot.pt`;
- one file ending in `_full.pt`;
- `seed_summary.csv`, `checkpoint_curves.csv`, `aggregate_summary.json`, and
  `multiseed_curves.npz`.

Do not commit `cloud_outputs/`, checkpoints, or generated experiment outputs.

---

## 14. Submit the full Experiment 1 run

Only proceed after the Batch smoke test succeeds and its outputs are present in
Cloud Storage. The approved full run uses five sequential seeds and includes
replay buffers in final checkpoints, so start conservatively with 64 GiB memory,
a 200 GiB boot disk, and a 96-hour cap:

```bash
export JOB_NAME="fhp-deep-cfr-exp1-$(date +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_deep_cfr_best_config_transfer.run \
    --output-root outputs/cloud/$JOB_NAME" \
  "n2-highmem-8" \
  "345600" \
  "8000" \
  "64000" \
  "200" \
  "pd-balanced"
```

This command does not change the approved training configuration. Machine,
memory, disk, timeout, and upload settings are execution resources rather than
algorithm hyperparameters.

To collect GNU `time` process statistics, prefix the Python command if the
Batch image provides `/usr/bin/time`:

```text
/usr/bin/time -v python -m experiments.fhp.exp1_deep_cfr_best_config_transfer.run ...
```

Its log includes elapsed time, CPU utilisation, and maximum resident set size.

---

## 15. Resource sizing

The wrapper requires the task's CPU and memory request to fit the selected VM.
Useful reference points are:

| Machine type | CPU milli | Memory MiB | Boot disk | Intended use |
| --- | ---: | ---: | ---: | --- |
| `n2-standard-4` | `4000` | `16000` | `100` GiB | Smoke test |
| `n2-standard-8` | `8000` | `32000` | `100`–`200` GiB | Intermediate profiling |
| `n2-highmem-8` | `8000` | `64000` | `200` GiB | Full starting point |

Availability, quotas, and exact RAM advertised for a machine family can vary
by region. Check before submission:

```bash
gcloud compute machine-types describe n2-highmem-8 \
  --zone europe-west1-b
```

The experiment uses a five-million-entry Python replay capacity and writes
buffer-inclusive checkpoints. Reduce resources only after measuring peak RAM,
checkpoint size, and runtime from a representative run. Do not alter replay
storage or omit buffers merely to fit a smaller VM without treating that as an
explicit experiment-protocol change.

Boot disk types accepted by the wrapper should be valid Compute Engine
persistent-disk types such as `pd-standard`, `pd-balanced`, or `pd-ssd`.
`pd-balanced` is the default.

---

## 16. Runtime limits and failure recovery

`MAX_RUN_SECONDS` is a hard task cap:

| Seconds | Duration |
| ---: | ---: |
| `3600` | 1 hour |
| `21600` | 6 hours |
| `43200` | 12 hours |
| `86400` | 24 hours |
| `259200` | 72 hours |
| `345600` | 96 hours |

The experiment writes a live `run_manifest.json`, per-seed snapshot manifests,
and policy snapshots during training. Completed seed outputs should therefore
survive a later-seed failure after the next periodic upload. The current runner
does not automatically resume an interrupted seed from its final checkpoint;
do not assume that resubmitting the same command continues where it stopped.

Use a new job name for every submission. Reusing a Cloud Storage prefix risks
mixing outputs from different code revisions or configurations.

---

## 17. Run a future experiment module

The wrapper is generic. Change only the module and output root:

```bash
export JOB_NAME="fhp-deep-cfr-expN-$(date +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.EXPERIMENT_PACKAGE.run \
    --output-root outputs/cloud/$JOB_NAME" \
  "MACHINE_TYPE" \
  "MAX_RUN_SECONDS" \
  "CPU_MILLI" \
  "MEMORY_MIB" \
  "BOOT_DISK_SIZE_GB" \
  "BOOT_DISK_TYPE"
```

The local module-existence check catches many copy/paste mistakes before a VM
is provisioned.

---

## 18. Clean up

Batch VMs are temporary and terminate after the job finishes. Keep failed job
records until debugging is complete. Then, if desired:

```bash
gcloud batch jobs delete "$JOB_NAME" \
  --location "$REGION" \
  --quiet
```

Inspect bucket contents before deleting anything:

```bash
gcloud storage ls --recursive "$BUCKET/"
```

Deleting cloud outputs is irreversible:

```bash
gcloud storage rm --recursive "$BUCKET/$JOB_NAME/"
```

Do not delete the bucket or service account while retained experiments still
depend on them.

---

## 19. Dependency-installation notes

The Batch helper creates an isolated environment at:

```text
/tmp/fhp-deep-cfr-venv
```

It installs CPU-only PyTorch wheels first, then the repository requirements,
the editable package, and `google-cloud-storage`. This avoids downloading CUDA
runtimes onto a CPU-only VM and avoids modifying the Python environment used by
the system Google Cloud CLI.

Uploads run through `scripts/upload_outputs_to_gcs.py` using that same virtual
environment. Local files under `outputs/...` are stored under:

```text
gs://BUCKET/JOB_NAME/outputs/...
```

If dependency installation fails, inspect the task logs for the resolved Git
commit, Python version, package resolver error, and free disk space before
changing versions or VM resources.

---

## 20. Quick troubleshooting checklist

Before asking Batch to retry, check:

1. `git push origin main` completed and `REPO_REF` exists remotely;
2. `REPO_URL` is cloneable without interactive credentials;
3. `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL` are set;
4. the user has `roles/batch.jobsEditor` and Service Account User;
5. the VM service account has Agent Reporter, Logs Writer, and bucket object
   access;
6. the chosen machine exists in the region and project quota is sufficient;
7. requested CPU and memory fit the selected machine;
8. the boot disk is large enough for dependencies and buffer checkpoints;
9. the smoke command passes locally and on Batch;
10. Cloud Storage contains the expected run manifest and checkpoints before a
    full run is considered successful.
