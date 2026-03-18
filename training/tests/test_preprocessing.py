import numpy as np
from sklearn.preprocessing import StandardScaler

from training.data.preprocessing import ScalerBundle


def test_set_fitted_scalers_allows_save_and_load(tmp_path):
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    Y = np.array([[0.1], [0.2]], dtype=np.float32)

    scaler_X = StandardScaler().fit(X)
    scaler_Y = StandardScaler().fit(Y)

    bundle = ScalerBundle(input_features=["a", "b"], target_features=["t"])
    bundle.set_fitted_scalers(scaler_X, scaler_Y)

    out = tmp_path / "scaler_bundle.json"
    bundle.save(str(out))
    loaded = ScalerBundle.load(str(out))

    assert loaded.input_features == ["a", "b"]
    assert loaded.target_features == ["t"]
    np.testing.assert_allclose(loaded.scaler_X.mean_, scaler_X.mean_)
    np.testing.assert_allclose(loaded.scaler_Y.mean_, scaler_Y.mean_)
