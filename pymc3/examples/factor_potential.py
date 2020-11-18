import pymc3 as pm

"""
You can add an arbitrary factor potential to the model likelihood using
pm.Potential. For example you can added Jacobian Adjustment using pm.Potential
when you do model reparameterization. It's similar to `target += u;` in
Stan.
"""


def build_model():
    with pm.Model() as model:
        x = pm.Normal("x", 1, 1)
        x2 = pm.Potential("x2", -(x ** 2))
    return model


def run(n=1000):
    model = build_model()
    if n == "short":
        n = 50
    with model:
        pm.sample(n)


if __name__ == "__main__":
    run()
