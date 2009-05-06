# -*- coding: UTF-8 -*-
"""NOTE: The code here has been altered. The docstring of the original package
follows.

Easy to use object-oriented thread pool framework.

A thread pool is an object that maintains a pool of worker threads to perform
time consuming operations in parallel. It assigns jobs to the threads
by putting them in a work request queue, where they are picked up by the
next available thread. This then performs the requested operation in the
background and puts the results in another queue.

The thread pool object can then collect the results from all threads from
this queue as soon as they become available or after all threads have
finished their work. It's also possible, to define callbacks to handle
each result as it comes in.

The basic concept and some code was taken from the book "Python in a Nutshell"
by Alex Martelli, copyright 2003, ISBN 0-596-00188-6, from section 14.5
"Threaded Program Architecture". I wrapped the main program logic in the
ThreadPool class, added the WorkRequest class and the callback system and
tweaked the code here and there. Kudos also to Florent Aide for the exception
handling mechanism.

Basic usage::

    >>> pool = ThreadPool(poolsize)
    >>> requests = makeRequests(some_callable, list_of_args, callback)
    >>> [pool.putRequest(req) for req in requests]
    >>> pool.wait()

See the end of the module code for a brief, annotated usage example.

Website : http://chrisarndt.de/projects/threadpool/


"""

__all__ = [
    'WorkRequest',
    'set_threadpool_size',
    'get_threadpool_size',
    '__PyMCThreadPool__',
    '__PyMCExcInfo__',
    '__PyMCLock__',
    'map_noreturn'
]

__author__ = "Christopher Arndt"
__version__ = "1.2.4"
__revision__ = "$Revision: 281 $"
__date__ = "$Date: 2008-05-04 17:41:41 +0200 (So, 04 Mai 2008) $"
__license__ = 'MIT license'


# standard library modules
import sys
import threading
import Queue
import traceback
import os

# exceptions
class NoResultsPending(Exception):
    """All work requests have been processed."""
    pass

class NoWorkersAvailable(Exception):
    """No worker threads available to process remaining requests."""
    pass


# internal module helper functions
def _handle_thread_exception(request, exc_info):
    """Default exception handler callback function.

    This just prints the exception info via ``traceback.print_exception``.

    """
    print exc_info
    traceback.print_exception(*exc_info)


# utility functions
def makeRequests(callable_, args_list, callback=None,
        exc_callback=_handle_thread_exception):
    """Create several work requests for same callable with different arguments.

    Convenience function for creating several work requests for the same
    callable where each invocation of the callable receives different values
    for its arguments.

    ``args_list`` contains the parameters for each invocation of callable.
    Each item in ``args_list`` should be either a 2-item tuple of the list of
    positional arguments and a dictionary of keyword arguments or a single,
    non-tuple argument.

    See docstring for ``WorkRequest`` for info on ``callback`` and
    ``exc_callback``.

    """
    requests = []
    for item in args_list:
        if isinstance(item, tuple):
            requests.append(
                WorkRequest(callable_, item[0], item[1], callback=callback,
                    exc_callback=exc_callback)
            )
        else:
            requests.append(
                WorkRequest(callable_, [item], None, callback=callback,
                    exc_callback=exc_callback)
            )
    return requests


# classes
class WorkerThread(threading.Thread):
    """Background thread connected to the requests/results queues.

    A worker thread sits in the background and picks up work requests from
    one queue and puts the results in another until it is dismissed.

    """

    def __init__(self, requests_queue, **kwds):
        """Set up thread in daemonic mode and start it immediatedly.

        ``requests_queue`` and ``results_queue`` are instances of
        ``Queue.Queue`` passed by the ``ThreadPool`` class when it creates a new
        worker thread.

        """
        threading.Thread.__init__(self, **kwds)
        self.setDaemon(1)
        self._requests_queue = requests_queue
        # self._results_queue = results_queue
        self._dismissed = threading.Event()
        self.start()

    def run(self):
        """Repeatedly process the job queue until told to exit."""
        while True:
            if self._dismissed.isSet():
                # we are dismissed, break out of loop
                break

            # get next work request.
            request = self._requests_queue.get()
            # print 'Worker thread %s running request %s' %(self, request)

            if self._dismissed.isSet():
                # we are dismissed, put back request in queue and exit loop
                self._requests_queue.put(request)
                break
            try:
                result = request.callable(*request.args, **request.kwds)
                if request.callback:
                    request.callback(request, result)
                del result
                self._requests_queue.task_done()
            except:
                request.exception = True
                if request.exc_callback:
                    request.exc_callback(request)
                self._requests_queue.task_done()
            finally:
                request.self_destruct()

    def dismiss(self):
        """Sets a flag to tell the thread to exit when done with current job."""
        self._dismissed.set()


class WorkRequest:
    """A request to execute a callable for putting in the request queue later.

    See the module function ``makeRequests`` for the common case
    where you want to build several ``WorkRequest`` objects for the same
    callable but with different arguments for each call.

    """

    def __init__(self, callable_, args=None, kwds=None, requestID=None,
            callback=None, exc_callback=_handle_thread_exception):
        """Create a work request for a callable and attach callbacks.

        A work request consists of the a callable to be executed by a
        worker thread, a list of positional arguments, a dictionary
        of keyword arguments.

        A ``callback`` function can be specified, that is called when the
        results of the request are picked up from the result queue. It must
        accept two anonymous arguments, the ``WorkRequest`` object and the
        results of the callable, in that order. If you want to pass additional
        information to the callback, just stick it on the request object.

        You can also give custom callback for when an exception occurs with
        the ``exc_callback`` keyword parameter. It should also accept two
        anonymous arguments, the ``WorkRequest`` and a tuple with the exception
        details as returned by ``sys.exc_info()``. The default implementation
        of this callback just prints the exception info via
        ``traceback.print_exception``. If you want no exception handler
        callback, just pass in ``None``.

        ``requestID``, if given, must be hashable since it is used by
        ``ThreadPool`` object to store the results of that work request in a
        dictionary. It defaults to the return value of ``id(self)``.

        """
        if requestID is None:
            self.requestID = id(self)
        else:
            try:
                self.str_requestID = requestID
                self.requestID = hash(requestID)
            except TypeError:
                raise TypeError("requestID must be hashable.")
        self.exception = False
        self.callback = callback
        self.exc_callback = exc_callback
        self.callable = callable_
        self.args = args or []
        self.kwds = kwds or {}

    def __str__(self):
        return "<WorkRequest id=%s>" % \
            (self.str_requestID)

    def self_destruct(self):
        """
        Avoids strange memory leak... for some reason the work request itself never
        gets let go, so if it has big arguments, or if its callable closes on big
        variables, there's trouble.
        """
        for attr in ['exception', 'callback', 'exc_callback', 'callable', 'args', 'kwds']:
            delattr(self, attr)

class ThreadPool:
    """A thread pool, distributing work requests and collecting results.

    See the module docstring for more information.

    """

    def __init__(self, num_workers, q_size=0, resq_size=0):
        """Set up the thread pool and start num_workers worker threads.

        ``num_workers`` is the number of worker threads to start initially.

        If ``q_size > 0`` the size of the work *request queue* is limited and
        the thread pool blocks when the queue is full and it tries to put
        more work requests in it (see ``putRequest`` method), unless you also
        use a positive ``timeout`` value for ``putRequest``.

        If ``resq_size > 0`` the size of the *results queue* is limited and the
        worker threads will block when the queue is full and they try to put
        new results in it.

        .. warning::
            If you set both ``q_size`` and ``resq_size`` to ``!= 0`` there is
            the possibilty of a deadlock, when the results queue is not pulled
            regularly and too many jobs are put in the work requests queue.
            To prevent this, always set ``timeout > 0`` when calling
            ``ThreadPool.putRequest()`` and catch ``Queue.Full`` exceptions.
        """
        self._requests_queue = Queue.Queue(q_size)
        # self._results_queue = Queue.Queue(resq_size)
        self.workers = []
        self.workRequests = {}
        self.createWorkers(num_workers)

    def createWorkers(self, num_workers):
        """Add num_workers worker threads to the pool.

        ``poll_timout`` sets the interval in seconds (int or float) for how
        ofte threads should check whether they are dismissed, while waiting for
        requests.

        """
        for i in range(num_workers):
            self.workers.append(WorkerThread(self._requests_queue))

    def dismissWorkers(self, num_workers):
        """Tell num_workers worker threads to quit after their current task."""
        for i in range(min(num_workers, len(self.workers))):
            worker = self.workers.pop()
            worker.dismiss()

    def setNumWorkers(self, num_workers):
        """Set number of worker threads to num_workers"""
        cur_num = len(self.workers)
        if cur_num > num_workers:
            self.dismissWorkers(cur_num - num_workers)
        else:
            self.createWorkers(num_workers - cur_num)

    def putRequest(self, request, block=True, timeout=0):
        """Put work request into work queue and save its id for later."""
        # don't reuse old work requests
        # print '\tthread pool putting work request %s'%request
        self._requests_queue.put(request, block, timeout)
        self.workRequests[request.requestID] = request



if os.environ.has_key('OMP_NUM_THREADS'):
    __PyMCThreadPool__ = ThreadPool(int(os.environ['OMP_NUM_THREADS']))
else:
    __PyMCThreadPool__ = ThreadPool(2)

class CountDownLatch(object):
    def __init__(self, n):
        self.n = n
        self.main_lock = threading.Lock()
        self.counter_lock = threading.Lock()
        self.main_lock.acquire()
    def countdown(self):
        self.counter_lock.acquire()
        self.n -= 1
        if self.n == 0:
            self.main_lock.release()
        self.counter_lock.release()
    def await(self):
        self.main_lock.acquire()
        self.main_lock.release()


def map_noreturn(targ, argslist):
    """
    parallel_call_noreturn(targ, argslist)

    :Parameters:
      - targ : function
      - argslist : list of tuples

    Does [targ(*args) for args in argslist] using the threadpool.
    """

    # Thanks to Anne Archibald's handythread.py for the exception handling mechanism.
    exceptions=[]
    n_threads = len(argslist)

    exc_lock = threading.Lock()
    done_lock = CountDownLatch(n_threads)

    def eb(wr, el=exc_lock, ex=exceptions, dl=done_lock):
        el.acquire()
        ex.append(sys.exc_info())
        el.release()

        dl.countdown()

    def cb(wr, value, dl=done_lock):
        dl.countdown()

    for args in argslist:
        __PyMCThreadPool__.putRequest(WorkRequest(targ, callback = cb, exc_callback=eb, args=args, requestID = id(args)))
    done_lock.await()

    if exceptions:
        a, b, c = exceptions[0]
        raise a, b, c


def set_threadpool_size(n):
    if n > 0:
        __PyMCThreadPool__.setNumWorkers(n)

def get_threadpool_size():
    return len(__PyMCThreadPool__.workers)

__PyMCLock__ = threading.Lock()
__PyMCExcInfo__ = [None]
