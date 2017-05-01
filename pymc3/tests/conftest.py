import theano

config = theano.configparser.change_flags(compute_test_value='raise')


def pytest_sessionstart(session):
    config.__enter__()


def pytest_sessionfinish(session, exitstatus):
    config.__exit__()
