"""Skip, rather than fail, tests that need the large rasters that are not stored in git (see REPRODUCE.md)."""
import pytest

_MISSING = ("No such file", "does not exist", "not found", "no such file")


def _is_missing_data(excinfo):
    text = "%s: %s" % (excinfo.typename, excinfo.value)
    return (excinfo.typename in ("RasterioIOError", "FileNotFoundError", "DriverError", "FileDataError") or "FileNotFoundError" in excinfo.typename or
            "Rasterio" in excinfo.typename or "rasterio" in text.lower()) and any(m in text for m in _MISSING)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.when in ("setup", "call") and _is_missing_data(call.excinfo):
        report.outcome = "skipped"
        report.longrepr = (str(item.fspath), item.location[1], "Skipped: needs downloaded data (see REPRODUCE.md)")
