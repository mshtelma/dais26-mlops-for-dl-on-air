def test_detector_signature_not_none():
    from src.serve.detector_pyfunc import build_signature_and_example
    sig, _example = build_signature_and_example()
    assert sig is not None
    assert sig.inputs is not None
    assert sig.outputs is not None


def test_embedder_signature_not_none():
    from src.serve.embedder_pyfunc import build_embedder_signature_and_example
    sig, _example = build_embedder_signature_and_example()
    assert sig is not None
    assert sig.inputs is not None
    assert sig.outputs is not None
