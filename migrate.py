"""One-shot Supabase migration runner - executes supabase/schema.sql over a
direct Postgres connection (psycopg2). DB password comes from SUPABASE_DB_PASSWORD
in backend/.env (never committed). Idempotent (schema.sql uses IF NOT EXISTS)."""
import sys
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parent
REF = "htkwjqxwsuwdwozbppxr"

def _db_password():
    env = ROOT / "backend" / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("SUPABASE_DB_PASSWORD="):
                return line.split("=", 1)[1].strip()
    import os
    return os.environ.get("SUPABASE_DB_PASSWORD", "")

PASSWORD = _db_password()
if not PASSWORD:
    sys.exit("Set SUPABASE_DB_PASSWORD=<your-db-password> in backend/.env "
             "(Supabase dashboard -> Settings -> Database). Never commit it.")

SQL = (ROOT / "supabase" / "schema.sql").read_text(encoding="utf-8")

CANDIDATES = [
    # direct connection (requires IPv6 support)
    dict(host=f"db.{REF}.supabase.co", port=5432, user="postgres"),
    # Supavisor pooler variants (region guessed - updated below if needed)
    dict(host=f"aws-0-ap-south-1.pooler.supabase.com", port=6543,
         user=f"postgres.{REF}"),
    dict(host=f"aws-0-us-east-1.pooler.supabase.com", port=6543,
         user=f"postgres.{REF}"),
    dict(host=f"aws-0-eu-central-1.pooler.supabase.com", port=6543,
         user=f"postgres.{REF}"),
    dict(host=f"aws-0-singapore-1.pooler.supabase.com", port=6543,
         user=f"postgres.{REF}"),
]

for cand in CANDIDATES:
    host = cand["host"]
    print(f"trying {host}:{cand['port']} ...", flush=True)
    try:
        conn = psycopg2.connect(
            host=host, port=cand["port"], user=cand["user"], dbname="postgres",
            password=PASSWORD, sslmode="require", connect_timeout=10)
        break
    except Exception as exc:                       # noqa: BLE001
        print("  failed:", str(exc)[:160])
else:
    sys.exit("all connection candidates failed")

print("connected ->", host)
with conn, conn.cursor() as cur:
    cur.execute(SQL)
    # verify
    cur.execute("""select table_name from information_schema.tables
                   where table_schema='public'
                   and table_name in ('scenes','jobs','matches','metrics')
                   order by table_name""")
    print("tables now present:", [r[0] for r in cur.fetchall()])
    cur.execute("select id from storage.buckets where id='raw-tiles'")
    print("bucket raw-tiles:", "ok" if cur.fetchone() else "missing")
conn.close()
print("MIGRATION COMPLETE")
