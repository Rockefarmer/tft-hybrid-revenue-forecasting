from tft_multihorizon.evaluation.metrics import mae, rmse, mape


def test_metrics_basic():
    y_true = [100, 200]
    y_pred = [110, 190]
    assert round(mae(y_true, y_pred), 4) == 10
    assert round(rmse(y_true, y_pred), 4) == 10
    assert round(mape(y_true, y_pred), 4) == 7.5
