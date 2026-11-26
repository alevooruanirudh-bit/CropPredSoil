from ml.suitability import remedies

def test_remedy_generation():
    gaps = [{"feature":"N","status":"low","severity":2.0},{"feature":"ph","status":"high","severity":1.5}]
    r = remedies(gaps)
    assert any("Nitrogen" in x["remedy"] for x in r)
    assert any("pH" in x["remedy"] for x in r)
