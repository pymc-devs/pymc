from aesara.graph.basic import ancestors
from aesara.tensor.random.op import RandomVariable


def assert_no_rvs(var):
    assert not any(
        isinstance(v.owner.op, RandomVariable) for v in ancestors([var]) if v.owner
    )
    return var
