import os


def pytest_sessionstart(session):
    os.environ["AESARA_FLAGS"] = ",".join(
        [
            os.environ.setdefault("AESARA_FLAGS", ""),
            "warn__ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise",
        ]
    )
