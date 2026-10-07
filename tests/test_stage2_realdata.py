"""Tests for the Stage-2 real-data modules (EIA tables shipped in data/real/eia)."""
import os

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from optigreen.data import eia, network as net, realcase as rc
from optigreen.forecasting import demand_model as dm
from optigreen.optimization import network_milp as nm
from optigreen.risk import outage_model as om

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "data", "real", "eia")


@pytest.fixture(scope="module")
def data():
    t = eia.load_tables(TABLES)
    return t, rc.demand_panel(t), rc.outage_labels(t)


def test_tables_present_and_plausible(data):
    t, panel, _ = data
    for name in ["demand_state_monthly", "demand_area_monthly", "district_monthly", "stocks_monthly",
                 "movements_monthly", "trade_monthly", "prices_monthly", "refinery_fuel_annual", "yields_padd_monthly"]:
        assert name in t and len(t[name]) > 100, name
    # 2019 U.S. gasoline sales ~ 8.2-9.3 million b/d -> 340-400 Mt per year
    gas19 = panel[(panel["product"] == "GAS") & (panel["month"].dt.year == 2019)]["kt"].sum() / 1000
    assert 340 < gas19 < 400
    # Colonial/Plantation corridor (PADD 3 -> PADD 1 by pipeline) ~ 2.2-3.0 million b/d in 2019
    mv = t["movements_monthly"]
    s = mv[(mv["from_area"] == "P3") & (mv["to_area"] == "P1") & (mv["mode"] == "pipeline")
           & (mv["month"].dt.year == 2019)].groupby("month")["kbbl"].sum().mean() / 30.4
    assert 2000 < s < 3200


def test_demand_panel_closes_to_area_totals(data):
    t, panel, _ = data
    ar = t["demand_area_monthly"]
    m = pd.Timestamp("2019-06-01")
    for area in ["1B", "P3"]:
        tot = ar[(ar["area"] == area) & (ar["month"] == m) & (ar["product"] == "GAS")]["kt"].iloc[0]
        mod = panel[(panel["area"] == area) & (panel["month"] == m) & (panel["product"] == "GAS")]["kt"].sum()
        assert abs(mod - tot) / tot < 0.02


@pytest.mark.parametrize("month,district", [("2005-09-01", "3C"), ("2008-09-01", "3B"), ("2017-09-01", "3B"),
                                            ("2019-07-01", "EC"), ("2021-02-01", "3B"), ("2021-09-01", "3C")])
def test_outage_labels_capture_known_events(data, month, district):
    _, _, lab = data
    row = lab[(lab["month"] == month) & (lab["district"] == district)]
    assert row["outage"].iloc[0] == 1, (month, district)


def test_refinery_co2_intensity_is_physical(data):
    t, _, _ = data
    em = rc.refinery_co2_intensity(t)
    us = em[(em["padd"] == "US") & (em["year"] == 2019)]["t_co2_per_kbbl"].iloc[0]
    assert 25 < us < 50  # kg CO2 per barrel of crude


def test_forecast_features_do_not_use_future(data):
    _, panel, _ = data
    o = pd.Timestamp("2019-03-01")
    sub = panel[panel["region"].isin(["TX", "NY", "POOL_P3"])]
    a = dm.build_rows(sub, [o])
    fut = sub.copy()
    fut.loc[fut["month"] >= o, "kt"] *= 10  # corrupt everything from the origin onwards
    b = dm.build_rows(fut, [o])
    pd.testing.assert_frame_equal(a[dm.FEATURES], b[dm.FEATURES])


def test_risk_features_do_not_use_future(data):
    _, _, lab = data
    o = pd.Timestamp("2017-08-01")
    a = om.build_rows(lab, [o])
    fut = lab.copy()
    cols = ["utilization_pct", "idio_dev", "outage", "severity"]
    fut.loc[fut["month"] >= o, cols] = fut.loc[fut["month"] >= o, cols].sample(frac=1, random_state=0).to_numpy()
    b = om.build_rows(fut, [o])
    pd.testing.assert_frame_equal(a[om.FEATURES], b[om.FEATURES])


def test_gat_graph_and_forward():
    A = om.district_graph()
    assert (A == A.T).all() and np.all(np.diag(A) == 1)
    import torch
    model = om.DistrictGAT(len(om.FEATURES))
    x = torch.randn(4, len(om.DISTRICTS), len(om.FEATURES))
    out = model(x, torch.tensor(A))
    assert out.shape == (4, len(om.DISTRICTS))


@pytest.fixture(scope="module")
def solved(data):
    t, panel, lab = data
    o = pd.Timestamp("2019-08-01")
    inp = nm.build_inputs(t, panel, None, o, labels=lab)   # actual demand (perfect information)
    m = nm.build_model(inp)
    status, el, gap = nm.solve(m, 120, 0.002)
    return inp, m, nm.extract(m, inp, status, el, gap)


def test_milp_is_mixed_integer_and_optimal(solved):
    inp, m, res = solved
    assert res.status == "optimal"
    assert res.n_int > 0  # integer tanker / barge / import cargoes
    assert res.kpis["fill_rate"] > 0.99


def test_milp_mass_balance_and_capacities(solved):
    inp, m, res = solved
    # district product balance: yield x crude == outflows
    for d in inp.districts.index:
        for t_ in inp.T:
            for k in eia.PRODUCTS:
                prod = inp.yields.loc[d, k] * pyo.value(m.x[d, t_])
                out = res.flows_dh[(res.flows_dh["district"] == d) & (res.flows_dh["product"] == k)
                                   & (res.flows_dh["month"] == inp.months[t_])]["kt"].sum()
                if len(res.flows_dr):
                    out += res.flows_dr[(res.flows_dr["district"] == d) & (res.flows_dr["product"] == k)
                                        & (res.flows_dr["month"] == inp.months[t_])]["kt"].sum()
                assert abs(prod - out) < 1e-3 * max(1, prod)
    # crude runs within effective capacity
    for r in res.crude.itertuples(index=False):
        cap = inp.districts.loc[r.district, "cap_kbcd"] * r.month.days_in_month * inp.districts.loc[r.district, "umax"]
        assert r.crude_kbbl <= cap * (1 + 1e-6)
    # inventories within bounds
    inv = res.inventory
    assert (inv["kt"] >= inv["floor"] - 1e-6).all() and (inv["kt"] <= inv["cap"] + 1e-6).all()


def test_recourse_evaluation_runs(data, solved):
    t, panel, lab = data
    inp, m, res = solved
    fs = nm.first_stage(m)
    avail = nm.actual_availability(inp, lab)
    ev = nm.evaluate(inp, fs, avail)
    assert ev.status == "optimal"
    assert 0.95 <= ev.kpis["fill_rate"] <= 1.0


def test_florida_is_supplied_by_sea(solved):
    """Florida has no interstate product pipeline: the plan must bring its product
    in by tanker to the Tampa hub, not at pipeline rates from Atlanta."""
    inp, m, res = solved
    Q = inp.arcs_hr
    fl = Q[Q["r"] == "FL"].set_index("h")
    truck = net.COST_ASSUMPTIONS["truck_usd_per_tkm"]
    for h, row in fl.iterrows():
        if h != "H1F":
            assert row["usd_per_t"] >= truck * row["km"]
    m0 = inp.months[0]
    fl_flow = res.flows_hr[(res.flows_hr["region"] == "FL") & (res.flows_hr["month"] == m0)]
    assert fl_flow[fl_flow["hub"] == "H1F"]["kt"].sum() > 0.8 * fl_flow["kt"].sum()
    marine = res.flows_dh[(res.flows_dh["lane"] == "P3>1C:marine") & (res.flows_dh["month"] == m0)]["kt"].sum()
    assert marine > 1000  # kt per month; EIA records about 2,000-2,600


def test_dashboard_runs_without_torch():
    """The deployed dashboard installs app/requirements.txt, which has no PyTorch:
    the risk module and the saved models must load without it."""
    import subprocess
    import sys
    code = (
        "import sys, gzip, pickle\n"
        # load the other libraries first: in this test environment they may probe an
        # installed torch themselves; the clean deployment environment has none at all
        "import xgboost, sklearn.linear_model, sklearn.metrics, sklearn.preprocessing\n"
        "for m in ('torch', 'torch.nn', 'torch.nn.functional'): sys.modules[m] = None\n"
        "sys.path.insert(0, 'src')\n"
        "import optigreen.risk.outage_model as om\n"
        "assert om.torch is None\n"
        "models = pickle.load(gzip.open('results/stage2/models.pkl.gz', 'rb'))\n"
        "assert set(models) >= {'forecasters', 'risk_by_year'}\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
