greek = ['alpha', 'eta', 'nu', 'tau',
     'beta', 'theta', 'xi', 'upsilon',
     'gamma', 'iota', 'phi', 'delta',
     'kappa', 'pi', 'chi', 'epsilon',
     'lambda', 'rho', 'psi', 'zeta',
     'mu', 'sigma', 'omega', 'Gamma',
     'Lambda', 'Sigma', 'Psi', 'Delta',
     'Xi', 'Upsilon', 'Omega', 'Theta', 'Pi']

class probability:
    def __init__(self, bra, ket=set(), pos='n'):
        """Create a probability instance p(bra|ket).
        pos: 'n' (numerator) or 'd' (denominator)
        """
        if type(bra)== str:
            bra = set([bra])
        if type(bra)!= set:
            self.bra = set(bra)
        else:
            self.bra = bra

        if type(ket) == str:
            ket = set([ket])
        if type(ket) != set:
            self.ket = set(ket)
        else:
            self.ket = ket

        self.pos = pos

    def __str__(self):
        s = 'p('
        for p in self.bra:
            s += p
            s += ','
        s = s[:-1]

        if len(self.ket) != 0:  # Not empty.
            s += '|'
            for p in self.ket:
                s += p
                s += ','
            s = s[:-1]
        s += ')'
        if self.pos == 'd':
            s += '^-1'
#            n = len(s)
#            s = n*'-' + '\n' + s
        return s

    def __eq__(self, p):
        if self.bra == p.bra and self.ket == p.ket and self.pos == p.pos:
            return True
        else:
            return False

    def inverse(self):
        if self.pos == 'n':
            self.pos = 'd'
        elif self.pos =='d':
            self.pos ='n'
        else:
            raise 'Bad attribute for pos.'

    def __rdiv__(self, x):
        if x != 1:
            raise 'Division by anything other than 1 is not allowed.'
        C = copy.copy(self)
        C.inverse()
        return C

    def shift(self, element):
        """Shift element from bra to ket, or ket to bra, depending on initial
        position."""
        element = self.as_set(element)
        for e in element:
            if e in self.bra:
                self.bra.remove(e)
                self.ket.add(e)
            elif e in self.ket:
                self.ket.remove(e)
                self.bra.add(e)
            else:
                raise AttributeError, 'No ' + e + ' in ' + self.__repr__() + '.'


    def as_set(self, x):
        """Return a set containing x."""
        if type(x) == str:
            return set([x])
        elif type(x) == set:
            return x
        else:
            return set(x)

    def reverse(self):
        """All kets become bras and all bras become kets."""
        if len(self.ket) == 0 :
            raise AttributeError, 'Cannot reverse, probability is not conditional.'
        self.ket, self.bra = self.bra, self.ket


class Bayes(probability):
    """Group multiple instances of probability."""
    def __init__(self, bra, ket = set()):
        self.group = []
        self.group.append(probability(bra, ket))


    def __str__(self):
        s = ''
        for g in self.group:
            s += g.__str__()
            s += ' '
        return s

    def latex(self):
        """Return the LaTeX description."""
        up = []
        down = []
        for g in self.group:
            if g.pos == 'n':
                up.append(g)
            if g.pos == 'd':
                down.append(g)

        if len(down) > 0:
            s = r'\frac{'
            for g in up:
                s += g.__str__()
            s += '}{'
            for g in down:
                s += g.__str__()[:-3]
            s += '}'
        else:
            s = ''
            for g in up:
                s += g.__str__()

        s = s.replace('|', r' \mid ')

        return s

    def shift(self, element, key = 0):
        """Use the product rule to shift a ket or bra."""
        element = self.as_set(element)

        g = self.group[key]
        bra = g.bra.intersection(element)
        ket = g.ket.intersection(element)
        cond = g.ket.difference(element)

        if len(bra) > 0:
            self.group.insert(key+1, probability(bra, cond))
            g.shift(bra)
        if len(ket) > 0:
            self.group.insert(key+2, probability(ket, cond, 'd'))
            g.shift(ket)

        self.clear_empty()
        self.clear_redundant()

    def clear_empty(self):
        to_clear = []
        for i, g in enumerate(self.group):
            if len(g.bra) == 0:
                to_clear.append(i)
        to_clear.sort()
        for i in to_clear[::-1]:
            self.group.pop(i)

    def clear_redundant(self):
        """Remove instances of a/a."""
        indicestoclear = []
        for i,g in enumerate(self.group):
            for j,g2 in enumerate(self.group[i+1:]):
                if g == 1/g2:
                    indicestoclear.append(i)
                    indicestoclear.append(j+i+1)
                elif g == g2:
                    indicestoclear.append(i)

        for i in indicestoclear[::-1]:
            self.group.pop(i)

# Example
p = Bayes(['alpha', 'beta'], ['x', 'y'])
p.shift('x');print p
p.shift('x', 0);print p
print '---'
print p
p.shift(['alpha', 'beta', 'y'])
print p
p.shift('alpha', 1)
print p
print p.latex()
