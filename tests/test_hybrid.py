from src.model.hybrid import get_hybrid


def test_get_hybrid():
    vit = get_hybrid('r50_vit_base_patch16_224')
    assert vit
