# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's
private reporting instead: **Security → Report a vulnerability** on this
repository. Include steps to reproduce and the affected version/commit.
You should get a reply within a week.

## Intended use and known limits

This is a **local, single-user portfolio project**, not a hardened product.

- **No authentication.** The API and dashboard have no login. Docker Compose
  therefore publishes every port on `127.0.0.1` only. Do not expose it to a
  network or the internet without adding authentication in front of it.
- **The Lab tab drives a ransomware *simulator*.** It is off unless
  `ENABLE_LAB=true`. It only works inside `./sandbox`, only touches files it
  created itself, and writes random bytes (no real encryption, no network).
- **Model files are pickles.** `joblib.load` executes code from the model file.
  Only load models you trained yourself with `python -m ml.train`.
- **Secrets** live only in `.env` (gitignored). Never commit it; rotate any key
  you suspect was exposed.
