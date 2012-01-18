__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

from pymc import *

def find_generations(stochastics):
    """
    A generation is the set of stochastic variables that only has parents in
    previous generations.
    """

    generations = []

    # Find root generation
    generations.append(set())
    all_children = set()

    for s in stochastics:
        all_children.update(s.extended_children & stochastics)
    generations[0] = stochastics - all_children

    # Find subsequent _generations
    children_remaining = True
    gen_num = 0
    while children_remaining:
        gen_num += 1


        # Find children of last generation
        generations.append(set())
        for s in generations[gen_num-1]:
            generations[gen_num].update(s.extended_children & stochastics)


        # Take away stochastics that have parents in the current generation.
        thisgen_children = set()
        for s in generations[gen_num]:
            thisgen_children.update(s.extended_children & stochastics)
        generations[gen_num] -= thisgen_children


        # Stop when no subsequent _generations remain
        if len(thisgen_children) == 0:
            children_remaining = False
    return generations


def ravel_submodel(stochastic_list):
    """
    Takes a list of stochastics and returns:
        - Indices corresponding to each,
        - Length of each,
        - Slices corresponding to each,
        - Total length,

    """

    N_stochastics = len(stochastic_list)
    stochastic_indices = []
    stochastic_len = np.zeros(N_stochastics, dtype=int)
    slices = np.zeros(N_stochastics, dtype=object)

    _len = 0
    for i in xrange(len(stochastic_list)):

        stochastic = stochastic_list[i]

        # Inspect shapes of all stochastics and create stochastic slices.
        if isinstance(stochastic.value, np.ndarray):
            stochastic_len[i] = len(np.ravel(stochastic.value))
        else:
            stochastic_len[i] = 1
        slices[i] = slice(_len, _len + stochastic_len[i])
        _len += stochastic_len[i]

        # Record indices that correspond to each stochastic.
        for j in xrange(len(np.ravel(stochastic.value))):
            stochastic_indices.append((stochastic, j))

    return stochastic_indices, stochastic_len, slices, _len

def set_ravelled_stochastic_values(vec, stochastics, slices):
    for stochastic in stochastics:
        stochastic.value = vec[slices[stochastic]].reshape(np.shape(stochastic.value))

def find_children_and_parents(stochastic_list):
    children = []
    parents = []
    for s in stochastic_list:
        if len(s.extended_children) > 0:
            if all([not child in stochastic_list for child in s.extended_children]):
                children.append(s)
        if all([not parent in stochastic_list for parent in s.extended_parents]):
            parents.append(s)

    return set(children), set(parents)

def order_stochastic_list(stochastics):

    generations = find_generations(stochastics)
    out = []
    for generation in generations[::-1]:
        out += list(generation)
    return out
