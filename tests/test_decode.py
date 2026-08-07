import glob
from pathlib import Path

import numpy as np
import pytest

from knee.dicom import decode_and_normalize

_SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"
_REAL_DCM_FILES = glob.glob(str(_SAMPLE_DIR / "**" / "*.dcm"), recursive=True)


@pytest.mark.skipif(not _REAL_DCM_FILES, reason="no real DICOM samples downloaded")
def test_decode_and_normalize_returns_correct_shape_dtype_and_range():
    result = decode_and_normalize(_REAL_DCM_FILES[0], size=224)

    assert result.shape == (224, 224)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


@pytest.mark.skipif(not _REAL_DCM_FILES, reason="no real DICOM samples downloaded")
def test_decode_and_normalize_uses_most_of_the_dynamic_range():
    # a real knee MRI slice, percentile-clipped to uint8, shouldn't collapse to a
    # near-constant image -- catches a broken normalization silently returning flat gray
    result = decode_and_normalize(_REAL_DCM_FILES[0], size=224)

    assert result.std() > 10


@pytest.mark.skipif(len(_REAL_DCM_FILES) < 5, reason="need several real DICOM samples")
def test_decode_and_normalize_works_across_multiple_real_files():
    for path in _REAL_DCM_FILES[:5]:
        result = decode_and_normalize(path, size=224)
        assert result.shape == (224, 224)
        assert result.dtype == np.uint8
