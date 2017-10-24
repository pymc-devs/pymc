import pymc3 as pm
with pm.Model() as model:
    a = pm.Gamma('a', mu=10.0, sd=2.0)
    b = pm.Gamma('b', mu=10.0, sd=2.0)

    trace = pm.sample(trace=[model.a, model.a_log__])
    assert len(trace.varnames) == 2
# you have to provide 4 vars:    
#    trace = pm.sample(trace=[model.a, model.a_log__, model.b, model.b_log__])

    pm.backends.text.dump('trace.text', trace)

    loaded = pm.backends.text.load('trace.text')
    print("loaded: ", loaded)
    x = loaded[0] #!!! Will throw a KeyError looking for 'b_log__'
