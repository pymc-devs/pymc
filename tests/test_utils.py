import aesara.tensor as at
import numpy as np
from aesara.graph.basic import Constant, ancestors
from aesara.tensor.random.basic import normal, uniform

from aeppl.abstract import MeasurableVariable
from aeppl.utils import change_rv_size, rvs_to_value_vars, walk_model
from tests.utils import assert_no_rvs


def test_change_rv_size():
    loc = at.as_tensor_variable([1, 2])
    rv = normal(loc=loc)
    assert rv.ndim == 1
    assert tuple(rv.shape.eval()) == (2,)

    rv_new = change_rv_size(rv, new_size=(3,), expand=True)
    assert rv_new.ndim == 2
    assert tuple(rv_new.shape.eval()) == (3, 2)

    # Make sure that the shape used to determine the expanded size doesn't
    # depend on the old `RandomVariable`.
    rv_new_ancestors = set(ancestors((rv_new,)))
    assert loc in rv_new_ancestors
    assert rv not in rv_new_ancestors

    rv_newer = change_rv_size(rv_new, new_size=(4,), expand=True)
    assert rv_newer.ndim == 3
    assert tuple(rv_newer.shape.eval()) == (4, 3, 2)

    # Make sure we avoid introducing a `Cast` by converting the new size before
    # constructing the new `RandomVariable`
    rv = normal(0, 1)
    new_size = np.array([4, 3], dtype="int32")
    rv_newer = change_rv_size(rv, new_size=new_size, expand=False)
    assert rv_newer.ndim == 2
    assert isinstance(rv_newer.owner.inputs[1], Constant)
    assert tuple(rv_newer.shape.eval()) == (4, 3)

    rv = normal(0, 1)
    new_size = at.as_tensor(np.array([4, 3], dtype="int32"))
    rv_newer = change_rv_size(rv, new_size=new_size, expand=True)
    assert rv_newer.ndim == 2
    assert tuple(rv_newer.shape.eval()) == (4, 3)

    rv = normal(0, 1)
    new_size = at.as_tensor(2, dtype="int32")
    rv_newer = change_rv_size(rv, new_size=new_size, expand=True)
    assert rv_newer.ndim == 1
    assert tuple(rv_newer.shape.eval()) == (2,)


def test_walk_model():
    d = at.vector("d")
    b = at.vector("b")
    c = uniform(0.0, d)
    c.name = "c"
    e = at.log(c)
    a = normal(e, b)
    a.name = "a"

    test_graph = at.exp(a + 1)
    res = list(walk_model((test_graph,)))
    assert a in res
    assert c not in res

    res = list(walk_model((test_graph,), walk_past_rvs=True))
    assert a in res
    assert c in res

    res = list(walk_model((test_graph,), walk_past_rvs=True, stop_at_vars={e}))
    assert a in res
    assert c not in res


def test_rvs_to_value_vars():

    a = at.random.uniform(0.0, 1.0)
    a.name = "a"
    a.tag.value_var = a_value_var = a.clone()

    b = at.random.uniform(0, a + 1.0)
    b.name = "b"
    b.tag.value_var = b_value_var = b.clone()

    c = at.random.normal()
    c.name = "c"
    c.tag.value_var = c_value_var = c.clone()

    d = at.log(c + b) + 2.0

    initial_replacements = {b: b_value_var, c: c_value_var}
    (res,), replaced = rvs_to_value_vars(
        (d,), initial_replacements=initial_replacements
    )

    assert res.owner.op == at.add
    log_output = res.owner.inputs[0]
    assert log_output.owner.op == at.log
    log_add_output = res.owner.inputs[0].owner.inputs[0]
    assert log_add_output.owner.op == at.add
    c_output = log_add_output.owner.inputs[0]

    # We make sure that the random variables were replaced
    # with their value variables
    assert c_output == c_value_var
    b_output = log_add_output.owner.inputs[1]
    assert b_output == b_value_var

    # There shouldn't be any `RandomVariable`s in the resulting graph
    assert_no_rvs(res)

    res_ancestors = list(walk_model((res,), walk_past_rvs=True))

    assert b_value_var in res_ancestors
    assert c_value_var in res_ancestors
    assert a_value_var not in res_ancestors


def test_rvs_to_value_vars_intermediate_rv():
    """Test that function replaces values above an intermediate RV."""
    a = at.random.uniform(0.0, 1.0)
    a.name = "a"
    a.tag.value_var = a_value_var = a.clone()

    b = at.random.uniform(0, a + 1.0)
    b.name = "b"
    b.tag.value_var = b.clone()

    c = at.random.normal()
    c.name = "c"
    c.tag.value_var = c_value_var = c.clone()

    d = at.log(c + b) + 2.0

    initial_replacements = {a: a_value_var, c: c_value_var}
    (res,), replaced = rvs_to_value_vars(
        (d,), initial_replacements=initial_replacements
    )

    # Assert that the only RandomVariable that remains in the graph is `b`
    res_ancestors = list(walk_model((res,), walk_past_rvs=True))

    assert (
        len(
            list(
                n
                for n in res_ancestors
                if n.owner and isinstance(n.owner.op, MeasurableVariable)
            )
        )
        == 1
    )

    assert c_value_var in res_ancestors
    assert a_value_var in res_ancestors
