# tools

A collection of utility sub-packages. Currently contains `google_cloud_storage_tools` for GCS access. See [`gcs_tools.md`](gcs_tools.md) for the full GCS API reference — this file covers the `tools` package structure and the key-resolution helper.

## Structure

```
packages/tools/
└── google_cloud_storage_tools/
    └── gcs_tools.py
```

## Import

```python
from packages.tools.google_cloud_storage_tools.gcs_tools import (
    gcs_json_key_file,
    list_files,
    read_file,
    save_file,
    write_file,
)
```

## Key Resolution — `gcs_json_key_file(key_file, secret_key)`

Resolves the GCP service account JSON key file path across environments in this order:

1. **Google Colab** — reads from `userdata.get(secret_key)`
2. **Kaggle** — reads from `UserSecretsClient().get_secret(secret_key)`
3. **RunPod** — reads from env var `RUNPOD_SECRET_<secret_key>` or `<secret_key>`
4. **Local** — expects `gcp_service_account_key.json` in the working directory

The secret value may be raw JSON or base64-encoded JSON — both are handled automatically.

All other functions (`list_files`, `read_file`, `save_file`, `write_file`) call this internally; no manual key setup is needed before using them.
