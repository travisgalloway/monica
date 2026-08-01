#!/usr/bin/env python3
"""Regenerate the tiny checked-in Parquet fixtures used by `monica-selfcheck`'s `MARK: parquet`
section (see swift/Sources/monica-selfcheck/main.swift) and by tests/test_swift_parquet.py.

Requires pyarrow (`.venv/bin/python swift/Fixtures/make-parquet-fixtures.py`, from the repo
root). Each fixture is a deliberately narrow probe of one corner of the pure-Swift reader
(swift/Sources/MonicaTokenizer/Parquet/) — see the #247 plan for the byte-level rationale.

    parquet-snappy-dict.parquet       7 rows, nulls, single row group, dictionary-encoded,
                                       snappy. This is the exact worked example in the #247
                                       plan's ParquetReader.swift section — the dictionary page
                                       body and data page body are decoded there byte-by-byte;
                                       keep this fixture's rows in sync with that worked example
                                       if you ever regenerate it.
    parquet-plain-multipage.parquet   PLAIN encoding (no dictionary) + multiple row groups, each
                                       split into multiple data pages by a small data_page_size.
                                       Exercises the page-walking loop beyond a single page/
                                       single row group — pyarrow buffers dictionary indices to
                                       row-group end, so this arm cannot be produced with
                                       dictionary encoding on (see the plan's open question #5).
    parquet-zstd.parquet              1 row, zstd — the negative path (unsupported codec).
"""
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).parent


def write(name: str, table: pa.Table, **kwargs) -> None:
    path = HERE / name
    pq.write_table(table, path, **kwargs)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    # --- parquet-snappy-dict.parquet -----------------------------------------------------
    # Dictionary build order = first-occurrence order of non-null values: "a" (row 0), ""
    # (row 2), "ccc" (row 3) -> dict ids 0/1/2. Matches the plan's worked example exactly.
    rows = ["a", None, "", "ccc", "a", None, ""]
    write("parquet-snappy-dict.parquet", pa.table({"text": rows}), compression="snappy")

    # --- parquet-plain-multipage.parquet -------------------------------------------------
    # 24 docs of ~250 B each, no dictionary, row_group_size=8 (-> 3 row groups) and a small
    # data_page_size so each row group spans several DATA_PAGE pages.
    docs = [f"doc {i:03d} " + ("x" * 250) for i in range(24)]
    write("parquet-plain-multipage.parquet", pa.table({"text": docs}),
          compression="snappy", use_dictionary=False, data_page_size=256, row_group_size=8)

    # --- parquet-zstd.parquet -- negative path: unsupported codec ------------------------
    write("parquet-zstd.parquet", pa.table({"text": ["only one row"]}), compression="zstd")

    # Print the actual row-group / page layout produced, so the numbers documented above (and
    # asserted in monica-selfcheck) are honest, not assumed.
    for name in ("parquet-snappy-dict.parquet", "parquet-plain-multipage.parquet"):
        pf = pq.ParquetFile(HERE / name)
        md = pf.metadata
        print(f"\n{name}: {md.num_row_groups} row group(s), {md.num_rows} row(s)")
        for rg in range(md.num_row_groups):
            col = md.row_group(rg).column(0)
            print(f"  row group {rg}: {col.num_values} values, encodings={list(col.encodings)}")


if __name__ == "__main__":
    main()
