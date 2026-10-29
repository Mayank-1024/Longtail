#!/usr/bin/env bash
# Download and unpack PandaSet into $LONGTAIL_DATA_DIR/pandaset/<sequence id>/.
#
# PandaSet is distributed after you accept its licence, so this script does not know the URL.
# Pass the archive URL you were given, or a local archive you already downloaded:
#
#   scripts/download_data.sh <url-or-path-to-archive> [sha256]
#
# Supports .zip, .tar, .tar.gz and .tgz. Re-running skips sequences that are already in place.

# ASSUMPTION A-01: dataset is PandaSet
# ASSUMPTION A-33: PandaSet sequences live at $LONGTAIL_DATA_DIR/pandaset/<sequence id>/

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

src="$1"
want_sha="${2:-}"
data_dir="${LONGTAIL_DATA_DIR:-./data}"
dest="$data_dir/pandaset"
work="$data_dir/downloads"
mkdir -p "$dest" "$work"

if [[ "$src" =~ ^https?:// ]]; then
  archive="$work/$(basename "${src%%\?*}")"
  echo "downloading to $archive (resumes if interrupted)"
  curl --fail --location --continue-at - --output "$archive" "$src"
else
  archive="$src"
  [[ -f "$archive" ]] || { echo "no such file: $archive" >&2; exit 1; }
fi

if [[ -n "$want_sha" ]]; then
  got_sha="$(shasum -a 256 "$archive" | cut -d' ' -f1)"
  [[ "$got_sha" == "$want_sha" ]] || { echo "sha256 mismatch: got $got_sha" >&2; exit 1; }
fi

extract="$(mktemp -d "$work/extract.XXXXXX")"
trap 'rm -rf "$extract"' EXIT
echo "extracting $archive"
case "$archive" in
  *.zip) unzip -q "$archive" -d "$extract" ;;
  *.tar | *.tar.gz | *.tgz) tar -xf "$archive" -C "$extract" ;;
  *) echo "unsupported archive type: $archive" >&2; exit 1 ;;
esac

# A sequence directory holds lidar/ and camera/; archives may nest them a few levels deep.
moved=0
skipped=0
while IFS= read -r lidar_dir; do
  seq_dir="$(dirname "$lidar_dir")"
  [[ -d "$seq_dir/camera" ]] || continue
  seq_id="$(basename "$seq_dir")"
  if [[ -e "$dest/$seq_id" ]]; then
    skipped=$((skipped + 1))
  else
    mv "$seq_dir" "$dest/$seq_id"
    moved=$((moved + 1))
  fi
done < <(find "$extract" -maxdepth 6 -type d -name lidar)

total="$(find "$dest" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
echo "added $moved sequences, skipped $skipped already present; $total under $dest"
[[ "$total" -gt 0 ]] || { echo "no sequences found in the archive" >&2; exit 1; }
