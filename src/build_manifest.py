"""
Run all dataset loaders, merge into one manifest, print EDA summary, save CSV.

Usage:
    python src/build_manifest.py
"""
import os
import csv
import collections
from config import DATA_PROCESSED_DIR, MANIFEST_CSV, EMOTIONS
from data_loaders import LOADERS


def build_manifest():
    all_rows = []
    print("Scanning datasets...\n")
    for name, loader_fn in LOADERS.items():
        rows = loader_fn()
        print(f"  {name:10s}: {len(rows):5d} usable clips found")
        if len(rows) == 0:
            print(f"    -> WARNING: no files found. Check that data/raw/{name}/ "
                  f"contains the extracted dataset.")
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo data found in any dataset folder. Nothing written.")
        return

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    with open(MANIFEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "emotion", "source_dataset", "actor_id", "gender"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nManifest written to {MANIFEST_CSV} ({len(all_rows)} total rows)\n")
    _print_eda(all_rows)


def _print_eda(rows):
    print("=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)

    by_source = collections.Counter(r["source_dataset"] for r in rows)
    print("\nClips per source dataset:")
    for src, count in sorted(by_source.items()):
        print(f"  {src:10s}: {count}")

    by_emotion = collections.Counter(r["emotion"] for r in rows)
    print("\nClips per emotion (unified label set):")
    for emo in EMOTIONS:
        print(f"  {emo:10s}: {by_emotion.get(emo, 0)}")

    speakers = set(r["actor_id"] for r in rows)
    print(f"\nTotal unique speakers: {len(speakers)}")

    # Cross-tab: which datasets contribute which emotions (helps explain class imbalance)
    print("\nEmotion coverage by dataset:")
    cross = collections.defaultdict(set)
    for r in rows:
        cross[r["source_dataset"]].add(r["emotion"])
    for src in sorted(cross.keys()):
        missing = set(EMOTIONS) - cross[src]
        print(f"  {src:10s}: covers {sorted(cross[src])}"
              + (f"  | missing: {sorted(missing)}" if missing else ""))
    print("=" * 60)


if __name__ == "__main__":
    build_manifest()
