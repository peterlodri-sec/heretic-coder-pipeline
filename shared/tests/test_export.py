from unittest.mock import MagicMock

from shared import export


def test_export_saves_merged_and_gguf():
    model, tok = MagicMock(), MagicMock()
    export.export_model(model, tok, "out_merged", "out_gguf")
    model.save_pretrained_merged.assert_called_once()
    model.save_pretrained_gguf.assert_called_once()
    assert model.save_pretrained_gguf.call_args.kwargs["quantization_method"] == "q4_k_m"
