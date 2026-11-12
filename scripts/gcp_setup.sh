#!/usr/bin/env bash
# =============================================================================
# AutoFix AI - GCP bootstrap
#
# Provisions (idempotently - safe to re-run):
#   1. Required APIs (run, artifactregistry, bigquery, aiplatform + IAM/WIF deps)
#   2. Artifact Registry Docker repo   : autofix-repo
#   3. BigQuery dataset + table        : autofix_db.telemetry_logs
#   4. Service accounts                : autofix-runtime (Cloud Run), autofix-deployer (CI)
#   5. Workload Identity Federation    : keyless GitHub Actions -> GCP
#
# Prerequisites:
#   - gcloud SDK installed (includes `bq`) and `gcloud auth login` done
#   - Billing enabled on the project
#   - Caller has Owner (or equivalent admin roles) on the project
#
# Usage:
#   PROJECT_ID=my-project GITHUB_REPO=my-org/autofix-ai ./scripts/gcp_setup.sh
#
# Optional env vars: REGION (default us-central1), BQ_LOCATION (default US)
# =============================================================================
set -euo pipefail

# ----------------------------- Configuration ---------------------------------
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-us-central1}"
BQ_LOCATION="${BQ_LOCATION:-US}"
GITHUB_REPO="${GITHUB_REPO:-}"            # format: owner/repo

AR_REPO="autofix-repo"
BQ_DATASET="autofix_db"
BQ_TABLE="telemetry_logs"

RUNTIME_SA_NAME="autofix-runtime"         # identity of the running Cloud Run service
DEPLOYER_SA_NAME="autofix-deployer"       # identity GitHub Actions impersonates
POOL_ID="github-pool"
PROVIDER_ID="github-provider"

# ------------------------------- Helpers -------------------------------------
log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# Retry a command: newly created service accounts take a few seconds to propagate.
retry() {
  local attempts=6 delay=5 n=1
  until "$@"; do
    if (( n >= attempts )); then
      die "Command failed after ${attempts} attempts: $*"
    fi
    echo "  ...not ready yet, retrying in ${delay}s (${n}/${attempts})"
    sleep "${delay}"
    n=$((n + 1))
  done
}

# ------------------------------ Preflight ------------------------------------
command -v gcloud >/dev/null 2>&1 || die "gcloud CLI not found. Install the Google Cloud SDK."
command -v bq     >/dev/null 2>&1 || die "bq CLI not found (ships with the Google Cloud SDK)."

[[ -n "${PROJECT_ID}" ]] || die "PROJECT_ID is not set and no default gcloud project is configured."
[[ -n "${GITHUB_REPO}" ]] || die "GITHUB_REPO is required, e.g. GITHUB_REPO=my-org/autofix-ai"
[[ "${GITHUB_REPO}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] \
  || die "GITHUB_REPO must look like 'owner/repo' (got '${GITHUB_REPO}')."

gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q . \
  || die "No active gcloud account. Run: gcloud auth login"

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA="${DEPLOYER_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

log "Project: ${PROJECT_ID} (#${PROJECT_NUMBER}) | Region: ${REGION} | BQ location: ${BQ_LOCATION}"
log "GitHub repo allowed to deploy: ${GITHUB_REPO}"

# ------------------------- 1. Enable required APIs ---------------------------
# The first four are the platform services. The remaining ones are required for
# service accounts, token exchange (WIF) and IAM policy changes.
log "Enabling APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  aiplatform.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project="${PROJECT_ID}"

# ------------------------ 2. Artifact Registry repo --------------------------
log "Ensuring Artifact Registry repo '${AR_REPO}' in ${REGION}..."
if gcloud artifacts repositories describe "${AR_REPO}" \
     --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  repo already exists - skipping"
else
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="AutoFix AI container images" \
    --project="${PROJECT_ID}"
fi

# ------------------------ 3. BigQuery dataset + table ------------------------
log "Ensuring BigQuery dataset '${BQ_DATASET}'..."
if bq --project_id="${PROJECT_ID}" show --format=none "${PROJECT_ID}:${BQ_DATASET}" >/dev/null 2>&1; then
  echo "  dataset already exists - skipping"
else
  bq --project_id="${PROJECT_ID}" --location="${BQ_LOCATION}" mk \
    --dataset \
    --description="AutoFix AI telemetry and diagnostics" \
    "${PROJECT_ID}:${BQ_DATASET}"
fi

log "Ensuring BigQuery table '${BQ_DATASET}.${BQ_TABLE}'..."
SCHEMA_FILE="$(mktemp)"
trap 'rm -f "${SCHEMA_FILE}"' EXIT

# Column names mirror the training features in train_model.py.
cat > "${SCHEMA_FILE}" <<'EOF'
[
  {"name": "event_timestamp",       "type": "TIMESTAMP", "mode": "REQUIRED", "description": "UTC time the reading was received"},
  {"name": "vehicle_id",            "type": "STRING",    "mode": "REQUIRED", "description": "Vehicle identifier (e.g. pseudonymised VIN)"},
  {"name": "session_id",            "type": "STRING",    "mode": "NULLABLE", "description": "Diagnostic session / request id"},
  {"name": "rpm",                   "type": "FLOAT",     "mode": "REQUIRED", "description": "PID 0x0C engine speed (rpm)"},
  {"name": "coolant_temp_c",        "type": "FLOAT",     "mode": "REQUIRED", "description": "PID 0x05 coolant temperature (C)"},
  {"name": "short_fuel_trim_pct",   "type": "FLOAT",     "mode": "REQUIRED", "description": "PID 0x06 short-term fuel trim (%)"},
  {"name": "long_fuel_trim_pct",    "type": "FLOAT",     "mode": "REQUIRED", "description": "PID 0x07 long-term fuel trim (%)"},
  {"name": "intake_pressure_kpa",   "type": "FLOAT",     "mode": "REQUIRED", "description": "PID 0x0B intake manifold pressure (kPa)"},
  {"name": "engine_load_pct",       "type": "FLOAT",     "mode": "REQUIRED", "description": "PID 0x04 calculated engine load (%)"},
  {"name": "predicted_fault",       "type": "STRING",    "mode": "NULLABLE", "description": "Normal | Fuel System Lean | Cooling Fault | Air Leak"},
  {"name": "fault_confidence",      "type": "FLOAT",     "mode": "NULLABLE", "description": "Model probability of predicted_fault (0-1)"},
  {"name": "model_version",         "type": "STRING",    "mode": "NULLABLE", "description": "Version from model.joblib bundle"},
  {"name": "repair_guide",          "type": "STRING",    "mode": "NULLABLE", "description": "LLM-generated repair steps"}
]
EOF

if bq --project_id="${PROJECT_ID}" show --format=none "${PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE}" >/dev/null 2>&1; then
  echo "  table already exists - skipping"
else
  # Day-partitioned for cheap time-bounded queries; clustered by vehicle for per-car lookups.
  bq --project_id="${PROJECT_ID}" mk \
    --table \
    --description="OBD-II telemetry, fault predictions and repair guides" \
    --time_partitioning_type=DAY \
    --time_partitioning_field=event_timestamp \
    --clustering_fields=vehicle_id,predicted_fault \
    "${PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE}" \
    "${SCHEMA_FILE}"
fi

# --------------------------- 4. Service accounts -----------------------------
ensure_sa() {
  local name="$1" display="$2" email="${1}@${PROJECT_ID}.iam.gserviceaccount.com"
  if gcloud iam service-accounts describe "${email}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "  ${email} already exists - skipping"
  else
    gcloud iam service-accounts create "${name}" \
      --display-name="${display}" --project="${PROJECT_ID}"
  fi
}

log "Ensuring service accounts..."
ensure_sa "${RUNTIME_SA_NAME}"  "AutoFix AI - Cloud Run runtime"
ensure_sa "${DEPLOYER_SA_NAME}" "AutoFix AI - GitHub Actions deployer"

grant_project_role() {  # $1 = member, $2 = role
  retry gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="$1" --role="$2" --condition=None --quiet >/dev/null
}

log "Granting least-privilege roles..."

# Runtime SA: call Gemini via Vertex AI, and write rows to the telemetry table only.
grant_project_role "serviceAccount:${RUNTIME_SA}" "roles/aiplatform.user"
retry bq --project_id="${PROJECT_ID}" add-iam-policy-binding \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/bigquery.dataEditor" \
  "${PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE}" >/dev/null

# Deployer SA: deploy Cloud Run revisions and push images to the one repo.
grant_project_role "serviceAccount:${DEPLOYER_SA}" "roles/run.admin"
retry gcloud artifacts repositories add-iam-policy-binding "${AR_REPO}" \
  --location="${REGION}" --project="${PROJECT_ID}" \
  --member="serviceAccount:${DEPLOYER_SA}" \
  --role="roles/artifactregistry.writer" --quiet >/dev/null

# Deployer may "act as" ONLY the runtime SA (required to deploy a service that runs as it).
retry gcloud iam service-accounts add-iam-policy-binding "${RUNTIME_SA}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${DEPLOYER_SA}" \
  --role="roles/iam.serviceAccountUser" --quiet >/dev/null

# -------------------- 5. Workload Identity Federation (GitHub) ---------------
log "Ensuring Workload Identity Pool '${POOL_ID}'..."
if gcloud iam workload-identity-pools describe "${POOL_ID}" \
     --location=global --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  pool already exists - skipping"
else
  gcloud iam workload-identity-pools create "${POOL_ID}" \
    --location=global \
    --display-name="GitHub Actions pool" \
    --project="${PROJECT_ID}"
fi

log "Ensuring OIDC provider '${PROVIDER_ID}'..."
if gcloud iam workload-identity-pools providers describe "${PROVIDER_ID}" \
     --workload-identity-pool="${POOL_ID}" \
     --location=global --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  provider already exists - skipping"
else
  # The attribute-condition is a hard gate: only tokens minted for THIS repo are
  # accepted, even if someone else learns the pool/provider names.
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_ID}" \
    --workload-identity-pool="${POOL_ID}" \
    --location=global \
    --display-name="GitHub OIDC provider" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner,attribute.ref=assertion.ref" \
    --attribute-condition="assertion.repository == '${GITHUB_REPO}'" \
    --project="${PROJECT_ID}"
fi

log "Allowing GitHub repo '${GITHUB_REPO}' to impersonate ${DEPLOYER_SA}..."
retry gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_SA}" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${GITHUB_REPO}" \
  --quiet >/dev/null

WIF_PROVIDER="$(gcloud iam workload-identity-pools providers describe "${PROVIDER_ID}" \
  --workload-identity-pool="${POOL_ID}" --location=global --project="${PROJECT_ID}" \
  --format='value(name)')"

# -------------------------------- Summary ------------------------------------
cat <<EOF

============================== SETUP COMPLETE ===============================
Artifact Registry : ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}
BigQuery table    : ${PROJECT_ID}.${BQ_DATASET}.${BQ_TABLE}
Runtime SA        : ${RUNTIME_SA}
Deployer SA       : ${DEPLOYER_SA}

Add these as GitHub repository variables/secrets (Settings > Secrets and variables > Actions):
  GCP_PROJECT_ID               = ${PROJECT_ID}
  GCP_REGION                   = ${REGION}
  GCP_WIF_PROVIDER             = ${WIF_PROVIDER}
  GCP_DEPLOYER_SERVICE_ACCOUNT = ${DEPLOYER_SA}
  GCP_RUNTIME_SERVICE_ACCOUNT  = ${RUNTIME_SA}

Example workflow auth step (job needs: permissions: { id-token: write, contents: read }):
  - uses: google-github-actions/auth@v2
    with:
      workload_identity_provider: \${{ vars.GCP_WIF_PROVIDER }}
      service_account: \${{ vars.GCP_DEPLOYER_SERVICE_ACCOUNT }}

Deploy with:  gcloud run deploy autofix-api --service-account=${RUNTIME_SA} ...
=============================================================================
EOF
