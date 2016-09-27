from pymc3 import *
import pandas as pd
from numpy.ma import masked_values

# Import data, filling missing values with sentinels (-999)
test_scores = pd.read_csv(get_data_file(
    'pymc3.examples', 'data/test_scores.csv')).fillna(-999)

# Extract variables: test score, gender, number of siblings, previous disability, age,
# mother with HS education or better, hearing loss identified by 3 months
# of age
(score, male, siblings, disability,
    age, mother_hs, early_ident) = test_scores[['score', 'male', 'siblings',
                                                'prev_disab', 'age_test',
                                                'mother_hs', 'early_ident']].astype(float).values.T

with Model() as model:

    # Impute missing values
    sib_mean = Exponential('sib_mean', 1)
    siblings_imp = Poisson('siblings_imp', sib_mean,
                           observed=masked_values(siblings, value=-999))

    p_disab = Beta('p_disab', 1, 1)
    disability_imp = Bernoulli(
        'disability_imp', p_disab, observed=masked_values(disability, value=-999))

    p_mother = Beta('p_mother', 1, 1)
    mother_imp = Bernoulli('mother_imp', p_mother,
                           observed=masked_values(mother_hs, value=-999))

    s = HalfCauchy('s', 5, testval=5)
    beta = Laplace('beta', 0, 100, shape=7, testval=.1)

    expected_score = (beta[0] + beta[1] * male + beta[2] * siblings_imp + beta[3] * disability_imp +
                      beta[4] * age + beta[5] * mother_imp + beta[6] * early_ident)

    observed_score = Normal(
        'observed_score', expected_score, s, observed=score)


with model:
    start = find_MAP()
    step1 = NUTS([beta, s, p_disab, p_mother, sib_mean], scaling=start)

    step2 = Metropolis([mother_imp.missing_values,
                        disability_imp.missing_values,
                        siblings_imp.missing_values])


def run(n=5000):
    if n == 'short':
        n = 100
    with model:
        trace = sample(n, [step1, step2], start)


if __name__ == '__main__':
    run()
