# Trivy exceptions

The `Trivy image scan` job fails on any HIGH or CRITICAL finding in the image or
the Dockerfile. `ignore-unfixed` is off, so a CVE with no patch available fails
the build too. That is deliberate: an unfixable CVE is a decision to record, not
one to configure away.

`.trivyignore.yaml` is the only way around a finding. Each entry needs:

- `id`: the finding exactly as Trivy reports it (`CVE-2026-12345`, `DS-0002`).
- `statement`: why it cannot be fixed here, in a sentence someone else can read.
  "wontfix" is not a reason.
- `expired_at`: a `YYYY-MM-DD` date at most 180 days out.

`scripts/check_trivy_exceptions.py` runs first in the scan job and fails it if
any entry is missing one of these, has expired, or runs past 180 days. Trivy stops
honouring an entry the day it expires, so a forgotten exception reopens the gate
rather than holding it shut.

Before adding one, check whether a newer base image already carries the fix. If
it does, bump the digest in the `Dockerfile` instead.

Run the check locally with:

```sh
python scripts/check_trivy_exceptions.py
```
