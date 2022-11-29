import textwrap

import aesara
import aesara.tensor as at
import pytest

from aeppl.printing import latex_pprint, pprint


def test_PreamblPPrinter():

    # Make sure we can print a `Function` and `FunctionGraph`
    mu = at.scalar("\\mu")
    sigma = at.scalar("\\sigma")
    b = at.scalar("b")

    y = b * at.random.normal(mu, sigma)

    y_fn = aesara.function([mu, sigma, b], y)

    expected = textwrap.dedent(
        r"""
    b in R, \mu in R, \sigma in R
    a ~ N(\mu, \sigma**2) in R
    (b * a)
    """
    )
    assert pprint(y_fn) == expected.strip()


def test_notex_print():

    normalrv_noname_expr = at.scalar("b") * at.random.normal(
        at.scalar("\\mu"), at.scalar("\\sigma")
    )
    expected = textwrap.dedent(
        r"""
    b in R, \mu in R, \sigma in R
    a ~ N(\mu, \sigma**2) in R
    (b * a)
    """
    )
    assert pprint(normalrv_noname_expr) == expected.strip()

    # Make sure the constant shape is show in values and not symbols.
    normalrv_name_expr = at.scalar("b") * at.random.normal(
        at.scalar("\\mu"), at.scalar("\\sigma"), size=[2, 1], name="X"
    )
    expected = textwrap.dedent(
        r"""
    b in R, \mu in R, \sigma in R
    X ~ N(\mu, \sigma**2) in R**(2 x 1)
    (b * X)
    """
    )
    assert pprint(normalrv_name_expr) == expected.strip()

    normalrv_noname_expr_2 = at.matrix("M") * at.random.normal(
        at.scalar("\\mu_2"), at.scalar("\\sigma_2")
    )
    normalrv_noname_expr_2 *= at.scalar("b") * at.random.normal(
        normalrv_noname_expr_2, at.scalar("\\sigma")
    ) + at.scalar("c")
    expected = textwrap.dedent(
        r"""
    M in R**(N^M_0 x N^M_1), \mu_2 in R, \sigma_2 in R
    b in R, \sigma in R, c in R
    a ~ N(\mu_2, \sigma_2**2) in R, d ~ N((M * a), \sigma**2) in R**(N^d_0 x N^d_1)
    ((M * a) * ((b * d) + c))
    """
    )
    assert pprint(normalrv_noname_expr_2) == expected.strip()

    expected = textwrap.dedent(
        r"""
    b in Z, c in Z, M in R**(N^M_0 x N^M_1)
    M[b, c]
    """
    )
    # TODO: "c" should be "1".
    assert (
        pprint(at.matrix("M")[at.iscalar("a"), at.constant(1, dtype="int")])
        == expected.strip()
    )

    expected = textwrap.dedent(
        r"""
    M in R**(N^M_0 x N^M_1)
    M[1]
    """
    )
    assert pprint(at.matrix("M")[1]) == expected.strip()

    expected = textwrap.dedent(
        r"""
    M in N**(N^M_0)
    M[2:4:0]
    """
    )
    assert pprint(at.vector("M", dtype="uint32")[0:4:2]) == expected.strip()


def test_tex_print():

    normalrv_noname_expr = at.scalar("b") * at.random.normal(
        at.scalar("\\mu"), at.scalar("\\sigma")
    )
    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
        b \in \mathbb{R}, \,\mu \in \mathbb{R}, \,\sigma \in \mathbb{R}
        \\
        a \sim \operatorname{N}\left(\mu, {\sigma}^{2}\right)\,  \in \mathbb{R}
        \\
        (b \odot a)
      \end{gathered}
    \end{equation}
    """
    )
    assert latex_pprint(normalrv_noname_expr) == expected.strip()

    normalrv_name_expr = at.scalar("b") * at.random.normal(
        at.scalar("\\mu"), at.scalar("\\sigma"), size=[2, 1], name="X"
    )
    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
        b \in \mathbb{R}, \,\mu \in \mathbb{R}, \,\sigma \in \mathbb{R}
        \\
        X \sim \operatorname{N}\left(\mu, {\sigma}^{2}\right)\,  \in \mathbb{R}^{2 \times 1}
        \\
        (b \odot X)
      \end{gathered}
    \end{equation}
    """
    )
    assert latex_pprint(normalrv_name_expr) == expected.strip()

    normalrv_noname_expr_2 = at.matrix("M") * at.random.normal(
        at.scalar("\\mu_2"), at.scalar("\\sigma_2")
    )
    normalrv_noname_expr_2 *= at.scalar("b") * at.random.normal(
        normalrv_noname_expr_2, at.scalar("\\sigma")
    ) + at.scalar("c")
    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
        M \in \mathbb{R}^{N^{M}_{0} \times N^{M}_{1}}
        \\
        \mu_2 \in \mathbb{R}, \,\sigma_2 \in \mathbb{R}
        \\
        b \in \mathbb{R}, \,\sigma \in \mathbb{R}, \,c \in \mathbb{R}
        \\
        a \sim \operatorname{N}\left(\mu_2, {\sigma_2}^{2}\right)\,  \in \mathbb{R}
        \\
        d \sim \operatorname{N}\left((M \odot a), {\sigma}^{2}\right)\,  \in \mathbb{R}^{N^{d}_{0} \times N^{d}_{1}}
        \\
        ((M \odot a) \odot ((b \odot d) + c))
      \end{gathered}
    \end{equation}
    """
    )
    assert latex_pprint(normalrv_noname_expr_2) == expected.strip()

    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
        b \in \mathbb{Z}, \,c \in \mathbb{Z}, \,M \in \mathbb{R}^{N^{M}_{0} \times N^{M}_{1}}
        \\
        M\left[b, \,c\right]
      \end{gathered}
    \end{equation}
    """
    )
    # TODO: "c" should be "1".
    assert (
        latex_pprint(at.matrix("M")[at.iscalar("a"), at.constant(1, dtype="int")])
        == expected.strip()
    )

    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
        M \in \mathbb{R}^{N^{M}_{0} \times N^{M}_{1}}
        \\
        M\left[1\right]
      \end{gathered}
    \end{equation}
    """
    )
    assert latex_pprint(at.matrix("M")[1]) == expected.strip()

    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
        M \in \mathbb{N}^{N^{M}_{0}}
        \\
        M\left[2:4:0\right]
      \end{gathered}
    \end{equation}
    """
    )
    assert latex_pprint(at.vector("M", dtype="uint32")[0:4:2]) == expected.strip()

    S_rv = at.random.invgamma(0.5, 0.5, name="S")
    T_rv = at.random.halfcauchy(1.0, name="T")
    Y_rv = at.random.normal(T_rv, at.sqrt(S_rv), name="Y")
    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
        T \sim \operatorname{C^{+}}\left(1, 1\right)\,  \in \mathbb{R}
        \\
        S \sim \operatorname{Gamma^{-1}}\left(0.5, 0.5\right)\,  \in \mathbb{R}
        \\
        Y \sim \operatorname{N}\left(T, {\sqrt{S}}^{2}\right)\,  \in \mathbb{R}
        \\
        Y
      \end{gathered}
    \end{equation}
    """
    )
    assert latex_pprint(Y_rv) == expected.strip()


@pytest.mark.xfail(
    reason=r"AePPL is not aware of the distributions' support and displays \mathbb{R} by default"
)
def test_tex_print_support_dimension():
    U_rv = at.random.uniform(0, 1, name="U")
    T_rv = at.random.halfcauchy(U_rv, name="T")
    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
        U \sim \operatorname{U}\left(1, 1\right)\,  \in \left[0, 1\right]
        \\
        T \sim \operatorname{C^{+}}\left(1, 1\right)\,  \in \mathbb{R}^{+}
        \\
        T
      \end{gathered}
    \end{equation}
    """
    )
    assert latex_pprint(T_rv) == expected.strip()
