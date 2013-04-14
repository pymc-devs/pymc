"""
A simple progress bar to monitor MCMC sampling progress.
Modified from original code by Corey Goldberg (2010)
"""

from __future__ import print_function

import sys, time
import uuid
try:
    from IPython.core.display import HTML, Javascript, display

    have_ipython = True
except ImportError:
    have_ipython = False

class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)

        self.start = time.time()
        self.last = 0
        if have_ipython:
            self.animate = self.animate_ipython
            self.divid = str(uuid.uuid4())
            self.sec_id = str(uuid.uuid4())

            pb = HTML(
                """
                <div style="float: left; border: 1px solid black; width:500px">
                  <div id="%s" style="background-color:blue; width:0%%">&nbsp;</div>
                </div> 
                <label id="%s" style="padding-left: 10px;" text = ""/>
                """ % (self.divid,self.sec_id))
            display(pb)
        else:
            self.animate = self.animate_noipython

    def animate_noipython(self, iter):
        if sys.platform.lower().startswith('win'):
            print(self, '\r', end='')
        else:
            print(self)
        self.update_iteration(iter)

    def animate_ipython(self, iter):
        elapsed = time.time() - self.start
        iter = iter + 1
        if elapsed - self.last > .5 or iter == self.iterations:
            self.last = elapsed

            self.update_iteration(iter)
            fraction = int(100*iter/float(self.iterations))

            display(Javascript("$('div#%s').width('%i%%')" % (self.divid, fraction)))
            display(Javascript("$('label#%s').text('%i%% in %.1f sec')" % (self.sec_id, fraction, round(elapsed, 1))))

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)


    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) / 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)
