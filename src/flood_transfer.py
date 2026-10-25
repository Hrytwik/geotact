"""T11: field-survey flood models carried to another city without retraining. In Bhubaneswar mode the logistic and random-forest
models are refitted on all 129 + 25 + 10,000 samples and saved; in another city they score every non-water municipal cell.
The transferred maps have no local flood truth and are reported as unvalidated. No torch here (LightGBM not used)."""
import pickle

import numpy as np

MODELS = ("logistic", "forest")


def save_models(seed=0):
    from src import flood_cnn_baselines as bl
    from src import flood_cnn_data as fd
    from src.config import CITY, HOME_OUTPUTS
    assert CITY == "bhubaneswar", "models are fitted on the Bhubaneswar field survey only"
    d = fd.build(seed)
    s = d["samples"]
    X, names = bl.features(d["stack"], s["rows"], s["cols"], fd.CHANNELS)
    models = {"logistic": bl.logistic_factory()(), "forest": bl.forest_factory(seed)()}
    for m in models.values():
        m.fit(X, s["y"])
    with open(HOME_OUTPUTS / "flood_cnn" / "transfer_models.pkl", "wb") as f:
        pickle.dump(dict(models=models, features=names), f)
    print(f"saved {list(models)} fitted on {len(s['y'])} samples ({int(s['y'].sum())} flood)", flush=True)


def score_city():
    from src import flood_cnn_baselines as bl
    from src import flood_cnn_data as fd
    from src import flood_cnn_r3 as r3
    from src import terrain as tr
    from src.config import HOME_OUTPUTS, OUTPUTS
    with open(HOME_OUTPUTS / "flood_cnn" / "transfer_models.pkl", "rb") as f:
        pack = pickle.load(f)
    stack, tf, crs, shape = fd.load_channels()
    inp = tr.real_inputs()
    ref = inp["bmc"] & ~inp["water"]
    means, centre = bl.local_means(stack), np.asarray(stack, dtype="float32")
    feats = lambda r, c: np.hstack([centre[:, r, c].T, means[:, r, c].T])
    out = OUTPUTS / "flood_cnn"
    out.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        m = pack["models"][name]
        sc = r3.score_map(ref, lambda r, c: m.predict_proba(feats(r, c))[:, 1])
        np.save(out / f"r3_scores_{name}.npy", sc)
        print(f"{name}: scored {int(np.isfinite(sc).sum())} cells (unvalidated transfer)", flush=True)


if __name__ == "__main__":
    from src.config import CITY
    from src.wardmap import memory_guard
    memory_guard(3.5)
    save_models() if CITY == "bhubaneswar" else score_city()
