from pathlib import Path
import yaml
from fed_finelabel_budget.cli import run_preflight

def test_offline_preflight_reaches_exact_budgets():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/p0.yaml").read_text(encoding="utf-8"))
    output = run_preflight(cfg)
    assert output["status"] == "PASS"
    assert output["fine_queries_observed"] == [20, 40, 80]
    assert output["remaining_queries"] == 0
    assert output["dataset_downloaded_or_read"] is False
