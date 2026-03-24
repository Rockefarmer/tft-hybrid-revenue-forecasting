from tft_hybrid.time_utils import parse_yq_str, yq_scalar


def test_parse_yq_str():
    assert parse_yq_str("2022Q4") == (2022, 4)
    assert parse_yq_str("2020 Q1") == (2020, 1)


def test_yq_scalar_ordering():
    assert yq_scalar(2020, 4) < yq_scalar(2021, 1)
