"""T11: the study area is a setting; Bhubaneswar stays the default, Cuttack gets its own folders (checked in fresh processes)."""
import json
import os
import subprocess
import sys


def _cfg(city=None):
    env = dict(os.environ)
    env.pop("GEOTACT_CITY", None)
    if city:
        env["GEOTACT_CITY"] = city
    code = "import json; from src import config as c; print(json.dumps(dict(city=c.CITY, bbox=c.BBOX, raw=str(c.RAW), out=str(c.OUTPUTS), home=str(c.HOME_OUTPUTS))))"
    return json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout)


def test_default_is_bhubaneswar_with_the_original_folders():
    c = _cfg()
    assert c["city"] == "bhubaneswar" and c["bbox"] == [85.70, 20.15, 85.93, 20.41]
    assert c["raw"].replace("\\", "/").endswith("data/raw") and c["out"].endswith("outputs") and c["home"] == c["out"]


def test_cuttack_has_its_own_box_and_folders_but_models_load_from_home():
    c = _cfg("cuttack")
    assert c["city"] == "cuttack" and c["bbox"] == [85.76, 20.40, 85.98, 20.56]
    assert "cuttack" in c["raw"] and "cuttack" in c["out"] and c["home"].endswith("outputs") and "cuttack" not in c["home"]


def test_validation_points_never_write_into_the_bhubaneswar_label_folder_for_another_city():
    env = dict(os.environ, GEOTACT_CITY="cuttack")
    code = "from src import validation_points as v; print(v.OUT, v.N_TOTAL, v.N_MIN, sep='|')"            # the path contains a space
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout.strip().split("|")
    assert out[0].endswith("validation_cuttack") and out[1] == "150" and out[2] == "20"
    env.pop("GEOTACT_CITY")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout.strip().split("|")
    assert out[0].replace("\\", "/").endswith("/validation") and out[1] == "300" and out[2] == "40"
