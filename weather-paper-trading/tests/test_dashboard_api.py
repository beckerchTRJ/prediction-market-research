from kwt import dashboard
from kwt.db import init_db


def test_structured_dashboard_payloads_support_empty_database(tmp_path):
    db = str(tmp_path / "kwt.db"); init_db(db)
    assert "coverage" in dashboard.overview(db)
    assert "fades" in dashboard.evidence(db)
    assert "cohorts" in dashboard.markouts(db)
    assert "summary" in dashboard.execution(db)
    assert "reserved_cost" in dashboard.risk(db)
    assert len(dashboard.cities(db)["cities"]) == 13


def test_dashboard_http_is_read_only_and_serves_static_app(tmp_path):
    db = str(tmp_path / "kwt.db"); init_db(db)
    assert dashboard.Handler.do_POST is dashboard.Handler._readonly
    assert dashboard.Handler.do_PUT is dashboard.Handler._readonly
    assert dashboard.Handler.do_PATCH is dashboard.Handler._readonly
    assert dashboard.Handler.do_DELETE is dashboard.Handler._readonly
    assert b"Edge Lab" in (dashboard.STATIC_DIR / "dashboard.html").read_bytes()


def test_json_cleaner_removes_non_finite_values():
    assert dashboard._clean({"x": float("nan"), "y": [float("inf")]}) == {
        "x": None, "y": [None]}


def test_execution_reports_share_of_cycles_at_capacity_cap(tmp_path):
    # "Time at cap" = share of quote cycles where a capacity limit
    # (max_capital_at_risk / max_net_inventory) blocked at least one order.
    # This is the real capacity-utilization number behind the block-rate ceiling.
    from kwt.db import connect
    db = str(tmp_path / "kwt.db"); init_db(db)
    c = connect(db)
    rows = [
        # cycle 1: capacity-blocked
        ("2026-07-19T00:00:00Z", "live", "KX-A", "bid", "blocked", "max_net_inventory"),
        ("2026-07-19T00:00:00Z", "live", "KX-A", "ask", "resting", None),
        # cycle 2: blocked, but not for capacity
        ("2026-07-19T00:05:00Z", "live", "KX-A", "bid", "blocked", "max_open_orders"),
        # cycle 3: capacity-blocked
        ("2026-07-19T00:10:00Z", "live", "KX-B", "ask", "blocked", "max_capital_at_risk"),
        # cycle 4: clean
        ("2026-07-19T00:15:00Z", "live", "KX-B", "ask", "resting", None),
    ]
    c.executemany("INSERT INTO live_orders (ts,mode,ticker,side,status,reason) "
                  "VALUES (?,?,?,?,?,?)", rows)
    c.commit(); c.close()

    s = dashboard.execution(db)["summary"]
    assert s["cycles"] == 4
    assert s["cycles_at_cap"] == 2
    assert s["at_cap_pct"] == 0.5


def test_at_cap_metric_is_null_on_an_empty_database(tmp_path):
    db = str(tmp_path / "kwt.db"); init_db(db)
    s = dashboard.execution(db)["summary"]
    assert s["cycles"] == 0 and s["cycles_at_cap"] == 0
    assert s["at_cap_pct"] is None
