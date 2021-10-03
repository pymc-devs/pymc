#tests are a work in progress (this is a draft)


def test_warning_msg():
    version = "1.2.3"
    reason = "Good reason"
    deprecated_args='x'

    @deprecator(version=version, reason=reason,deprecated_args='x')
    def foo(x,y):
        pass


    # you can also use with pytest.warns(DeprecationWarning, match="some part of the message"):
    # ^michael
    
    with warnings.catch_warnings(record=True) as warns:
        foo()
    warn = warns[0]
    assert version in str(warn.message)
    assert reason in str(warn.message)
    assert deprecated_args in str(warn.message)


def test_ignore_action():
    @deprecator(reason='some reason', version='1', action='ignore')
    def foo():
        pass

    with warnings.catch_warnings(record=True) as warns:
        foo()
    assert len(warns) == 0


def test_sphinx_docstring():
    @deprecator(version="1.0", reason="some reason")
    def basicfunc(x,y):
        """
        something
        """
        return x+y
    assert basicfunc.__doc__=='\nsomething\n\n.. deprecated:: 1.0\n   some reason\n'
