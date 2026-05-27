from __future__ import annotations

import base64
import json

import pytest

from dais26_dentex.drift.inference_table_reader import parse_request_payload


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# Fixture 1: valid dataframe_split with 1 image
def test_dataframe_split_single_image() -> None:
    img_bytes = b"\x89PNG\r\nfake_image_data"
    payload = json.dumps(
        {
            "dataframe_split": {
                "columns": ["image", "metadata"],
                "data": [[_b64(img_bytes), "some_meta"]],
            }
        }
    )
    result = parse_request_payload(payload)
    assert len(result) == 1
    assert result[0] == img_bytes


# Fixture 2: valid dataframe_records with 2 images
def test_dataframe_records_two_images() -> None:
    img1 = b"fake_image_1"
    img2 = b"fake_image_2"
    payload = json.dumps(
        {
            "dataframe_records": [
                {"image": _b64(img1), "label": "a"},
                {"image": _b64(img2), "label": "b"},
            ]
        }
    )
    result = parse_request_payload(payload)
    assert len(result) == 2
    assert result[0] == img1
    assert result[1] == img2


# Fixture 3: None input (1 MiB cap — payload was dropped)
def test_none_input_returns_empty() -> None:
    result = parse_request_payload(None)
    assert result == []


# Fixture 4: malformed JSON
def test_malformed_json_returns_empty() -> None:
    result = parse_request_payload("not json")
    assert result == []


# Fixture 5: invalid base64 value — returns [] and logs a warning
def test_invalid_base64_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    payload = json.dumps(
        {
            "dataframe_records": [
                {"image": "not_base64!!!"},
            ]
        }
    )
    import logging

    with caplog.at_level(logging.WARNING, logger="dais26_dentex.drift.inference_table_reader"):
        result = parse_request_payload(payload)
    assert result == []
    assert any("Invalid base64" in msg for msg in caplog.messages)


# Fixture 6: unknown schema — returns [] and logs a warning
def test_unknown_schema_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    payload = json.dumps({"foo": "bar"})
    import logging

    with caplog.at_level(logging.WARNING, logger="dais26_dentex.drift.inference_table_reader"):
        result = parse_request_payload(payload)
    assert result == []
    assert any("Unknown request schema" in msg for msg in caplog.messages)
