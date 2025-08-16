import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"

def main():
    if not ENV.exists():
        print(".env not found; nothing to clean.")
        return
    lines = ENV.read_text(encoding="utf-8").splitlines()
    pairs = {}
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", ln)
        if m:
            k, v = m.group(1), m.group(2)
            v = v.strip()
            pairs[k] = v
    # Remove obviously unused keys from legacy setups
    unused = {"USE_ENV_CONFIG"}  # keep minimal; code defaults already handle this
    for k in unused:
        pairs.pop(k, None)
    # Pretty print sorted keys
    key_order = [
        "FMP_API_KEY",
        "OPENAI_API_KEY", "OPENAI_MODEL",
        "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT",
        "NEWS_API_KEY",
        "SENTRY_DSN", "SLACK_WEBHOOK_URL",
    ]
    # Merge known keys first, then any extras
    ordered = []
    seen = set()
    for k in key_order:
        if k in pairs:
            ordered.append((k, pairs[k]))
            seen.add(k)
    for k in sorted([x for x in pairs.keys() if x not in seen]):
        ordered.append((k, pairs[k]))

    out = []
    out.append("# VP Investments environment (cleaned)\n")
    for k, v in ordered:
        out.append(f"{k}={v}")
    ENV.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Cleaned .env with {len(ordered)} keys.")

if __name__ == "__main__":
    main()
