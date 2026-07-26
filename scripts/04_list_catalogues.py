"""

USAGE
-----
Search for collections about a topic:
    uv run python scripts\\04_list_catalogues.py retail
    uv run python scripts\\04_list_catalogues.py household spending
    uv run python scripts\\04_list_catalogues.py trade

Show only currently-active collections (hide discontinued ones):
    uv run python scripts\\04_list_catalogues.py retail --active

Look up one catalogue number directly:
    uv run python scripts\\04_list_catalogues.py --cat 5206.0

WHAT TO DO WITH THE RESULT
--------------------------
1. Find the collection that covers your topic and is not CEASED.
2. Put its catalogue number into the SeriesSpec in src/ausgdp/config.py.
3. Re-run scripts/01_discover.py to list the series inside it.
"""

from __future__ import annotations

import sys

import pandas as pd


def main(argv: list[str]) -> None:
    try:
        import readabs as ra
    except ImportError:
        sys.exit("readabs not installed. Run:  uv sync")

    flags = {a for a in argv if a.startswith("--")}
    words = [a for a in argv if not a.startswith("--")]

    print("Downloading the ABS catalogue directory (cached after first run)...")
    try:
        cat = ra.abs_catalogue()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Could not fetch the ABS catalogue: {type(exc).__name__}: {exc}")

    cat = cat.reset_index()
    id_col = cat.columns[0]
    cat = cat.rename(columns={id_col: "Catalogue"})

    print(f"{len(cat)} collections listed by the ABS.\n")

    # Direct lookup of one catalogue number
    if "--cat" in flags:
        if not words:
            sys.exit("Give a catalogue number, e.g. --cat 5206.0")
        target = words[0]
        hit = cat[cat["Catalogue"].astype(str).str.strip() == target]
        if hit.empty:
            print(f"Catalogue {target} is NOT in the current ABS directory.")
            print("It has probably been discontinued or renamed.")
            print("Try searching instead:  ...\\04_list_catalogues.py <keyword>")
            return
        _show(hit)
        return

    if not words:
        sys.exit(
            "Give a search word, e.g.\n"
            "    uv run python scripts\\04_list_catalogues.py retail"
        )

    # Search across every text column
    text = cat.astype(str).agg(" | ".join, axis=1)
    mask = pd.Series(True, index=cat.index)
    for w in words:
        mask &= text.str.contains(w, case=False, na=False, regex=False)

    hits = cat[mask]

    if "--active" in flags and "Status" in hits.columns:
        hits = hits[~hits["Status"].astype(str).str.contains("CEASED", case=False, na=False)]

    label = " ".join(words)
    print(f"Searching for: {label}")
    print(f"{len(hits)} matching collections\n")

    if hits.empty:
        print("Nothing found. Try a single, more general word ('trade', 'spending').")
        return

    _show(hits)

    print("\nPick the collection that is NOT ceased, put its catalogue number")
    print("into src/ausgdp/config.py, then re-run scripts\\01_discover.py")


def _show(hits: pd.DataFrame) -> None:
    wanted = ["Catalogue", "Topic", "Parent Topic", "Theme", "Status"]
    cols = [c for c in wanted if c in hits.columns]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.max_rows", 80)
    print(hits[cols].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
