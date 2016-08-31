"""
A simple progress bar to monitor MCMC sampling progress.
Modified from original code by Corey Goldberg (2010)
"""

from __future__ import print_function

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import sys
import time
import uuid

__all__ = ['progress_bar']


class ProgressBar(object):
    def __init__(self, iterations, animation_interval=.5):
        self.iterations = iterations
        self.start = time.time()
        self.last = 0
        self.animation_interval = animation_interval

    def percentage(self, i):
        return 100 * i / float(self.iterations)

    def update(self, i):
        elapsed = time.time() - self.start
        i += 1

        if elapsed - self.last > self.animation_interval:
            self.animate(i + 1, elapsed)
            self.last = elapsed
        elif i == self.iterations:
            self.animate(i, elapsed)


class TextProgressBar(ProgressBar):
    def __init__(self, iterations, printer):
        self.fill_char = '-'
        self.width = 40
        self.printer = printer

        super(TextProgressBar, self).__init__(iterations)
        self.update(0)

    def animate(self, i, elapsed):
        self.printer(self.progbar(i, elapsed))

    def progbar(self, i, elapsed):
        bar = self.bar(self.percentage(i))
        return "[%s] %i of %i complete in %.1f sec" % (bar, i, self.iterations, round(elapsed, 1))

    def bar(self, percent):
        all_full = self.width - 2
        num_hashes = int(percent / 100 * all_full)

        bar = self.fill_char * num_hashes + ' ' * (all_full - num_hashes)

        info = '%d%%' % percent
        loc = (len(bar) - len(info)) // 2
        return replace_at(bar, info, loc, loc + len(info))


def replace_at(str, new, start, stop):
    return str[:start] + new + str[stop:]


def consoleprint(s):
    if sys.platform.lower().startswith('win'):
        print(s, '\r', end='')
    else:
        print(s)


def ipythonprint(s):
    print('\r', s, end='')
    sys.stdout.flush()


class IPythonNotebookPB(ProgressBar):
    def __init__(self, iterations):
        if not hasattr(self, '_widget'):
            from IPython.html import widgets
            from IPython.display import display

            self._widget = widgets.FloatProgress()
            display(self._widget)
            self._widget.value = 0

        super(IPythonNotebookPB, self).__init__(iterations)

    def animate(self, i, elapsed):
        percentage = int(self.percentage(i))

        # Calculate percent completion, and update progress bar
        self._widget.value = percentage
        self._widget.description = ' ({0:>3s}%)'.format('{:d}'.format(percentage))


def run_from_ipython():
        try:
            __IPYTHON__
            return True
        except NameError:
            return False


def progress_bar(iters):
    try:
        from IPython.html import widgets
        widgets.FloatProgress()
        return IPythonNotebookPB(iters)
    except RuntimeError:
        return TextProgressBar(iters, ipythonprint)
    except ImportError:
        return TextProgressBar(iters, consoleprint)
