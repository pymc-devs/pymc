if __name__ == '__main__':  # Avoid loading during tests.

    import pymc3 as pm
    import pymc3.examples.hierarchical as hier

    with hier.model:
        trace = pm.sample(3000, hier.step, hier.start, trace='sqlite')
