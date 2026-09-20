from kwt.db import connect


def test_live_fill_markouts_table_exists_with_expected_columns():
    conn = connect(":memory:")
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(live_fill_markouts)")}
    assert cols == {"fill_id", "ticker", "created_time", "direction",
                    "mid_at_fill", "mo_15", "mo_30", "mo_60", "mo_settle", "computed_ts"}
