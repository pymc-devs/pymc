import mcex as mx


random.seed(1)

n_groups = 2
no_pergroup = 30 
n_observed = no_pergroup * n_groups
n_group_predictors = 1
n_predictors = 3

groups = np.concatenate([ [i]*no_pergroup for i in range(n_groups)])
group_predictors = ones((n_groups, 1)) #random.normal(size = (n_groups, n_group_predictors))
predictors       = random.normal(size = (n_observed, n_predictors))

group_effects = random.normal( size = (group_predictors))
effects = random.normal(size = (n_groups, predictors)) + sum(group_effects[newaxis, :] * group_predictors, 1)

y = sum(effects[group, :] * predictors, 1) + random.normal(size = (n_observed))

model = Model()

#m_g ~ N(0, .1)
m_g = FreeVariable("m_g", (nx, n_pg), float)
AddVar(model, m_g, Normal(0, .1))


# sg ~ 
sg = 1
#m ~ N(mg * pg, sg)
m = FreeVariable("m", (ng, nx), float)
AddVar(model, m, Normal( dot(mg ,  pg),sg))

#s ~ 
s = 10
#y ~ Normal(m[g] * p, s)
AddVar(model, y, Normal( dot(m[g,:], p),s))



