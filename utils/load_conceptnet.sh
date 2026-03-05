#!/usr/bin/env bash
# Load ConceptNet 5.7 English-only edges into PostgreSQL.
#
# Usage: ./utils/load_conceptnet.sh /path/to/conceptnet-assertions-5.7.0.csv.gz
#
# Filters to English-to-English Synonym, SimilarTo, IsA, and PartOf edges
# only (~5% of the full file). Indexes follow the load.
#
# Run time: a few minutes for the awk filter + COPY; index build adds a few more.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/conceptnet-assertions-5.7.0.csv.gz" >&2
    exit 1
fi

CSV_GZ="$1"

if [ ! -f "$CSV_GZ" ]; then
    echo "File not found: $CSV_GZ" >&2
    exit 1
fi

# Load .env from project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

: "${DB_HOST:?DB_HOST not set}"
: "${DB_USER:?DB_USER not set}"
: "${DB_PASSWORD:?DB_PASSWORD not set}"
: "${DB_NAME:?DB_NAME not set}"

export PGPASSWORD="$DB_PASSWORD"

echo "Truncating conceptnet_edges..."
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "TRUNCATE TABLE conceptnet_edges;"

echo "Loading from $CSV_GZ (filtering to English Synonym/SimilarTo/IsA/PartOf)..."
echo "This may take several minutes..."

zcat "$CSV_GZ" \
    | awk -F'\t' '
        function norm(uri,    arr, n) {
            n = split(uri, arr, "/")
            return (n >= 4) ? ("/" arr[2] "/" arr[3] "/" arr[4]) : uri
        }
        ($2=="/r/Synonym"||$2=="/r/SimilarTo"||$2=="/r/IsA"||$2=="/r/PartOf") &&
        $3 ~ /^\/c\/en\// &&
        $4 ~ /^\/c\/en\// {
            s = norm($3); e = norm($4)
            if (s != e) print $2 "\t" s "\t" e "\t1.0"
        }' \
    | sort -u \
    | psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
           -c "COPY conceptnet_edges (relation, start_uri, end_uri, weight) FROM STDIN (FORMAT text, DELIMITER E'\t')"

echo "Building indexes..."
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" << 'SQL'
    CREATE INDEX IF NOT EXISTS idx_cn_start_rel ON conceptnet_edges (start_uri, relation);
    CREATE INDEX IF NOT EXISTS idx_cn_end_rel   ON conceptnet_edges (end_uri,   relation);
    CREATE INDEX IF NOT EXISTS idx_cn_rel       ON conceptnet_edges (relation);
SQL

echo "Done. Row count:"
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
     -c "SELECT relation, COUNT(*) FROM conceptnet_edges GROUP BY relation ORDER BY count DESC;"
