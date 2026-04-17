import sys
import types

import numpy as np
import pandas as pd

from training.data.simc_dataset import SIMCDataset


class _FakeTree:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def keys(self):
        return list(self._frame.columns)

    def arrays(self, wanted, library="pd"):
        assert library == "pd"
        return self._frame[wanted].copy()


class _FakeRootFile:
    def __init__(self, frame: pd.DataFrame):
        self._tree = _FakeTree(frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getitem__(self, name: str):
        if name != "h10":
            raise KeyError(name)
        return self._tree


class _FakeUproot(types.SimpleNamespace):
    def __init__(self, frame: pd.DataFrame):
        super().__init__(open=lambda _: _FakeRootFile(frame))


def test_simc_dataset_resolves_ps_aliases_and_fry(monkeypatch):
    n = 6
    df = pd.DataFrame(
        {
            "psxfp": np.linspace(-1.0, 1.0, n, dtype=np.float32),
            "psyfp": np.linspace(-2.0, 2.0, n, dtype=np.float32),
            "psxpfp": np.linspace(-0.1, 0.1, n, dtype=np.float32),
            "psypfp": np.linspace(-0.2, 0.2, n, dtype=np.float32),
            "psdeltai": np.linspace(-5.0, 5.0, n, dtype=np.float32),
            "psxptari": np.linspace(-0.03, 0.03, n, dtype=np.float32),
            "psyptari": np.linspace(-0.02, 0.02, n, dtype=np.float32),
            "psztari": np.linspace(-10.0, 10.0, n, dtype=np.float32),
            "fry": np.linspace(-0.5, 0.5, n, dtype=np.float32),
        }
    )

    monkeypatch.setitem(sys.modules, "uproot", _FakeUproot(df))

    ds = SIMCDataset(
        root_file_paths=["dummy.root"],
        feature_schema=["x_fp", "y_fp", "xp_fp", "yp_fp", "fry"],
        include_fry=True,
        fit_scalers=False,
    )

    assert len(ds) == n
    assert ds.X.shape == (n, 5)
    assert ds.Y.shape == (n, 4)
    assert ds.feature_names == ["x_fp", "y_fp", "xp_fp", "yp_fp", "fry"]

    first = ds[0]
    assert set(first["targets"].keys()) == {"delta", "xptar", "yptar", "ytar"}
