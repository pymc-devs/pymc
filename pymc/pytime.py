"""
Determine function execution time.

Connelly Barnes.  Public domain.

 >>> def f():
 ... return sum(range(10))
 ...
 >>> pytime(f)
 (Time to execute function f, including function call overhead).
 >>> 1.0/pytime(f)
 (Function calls/sec, including function call overhead).
 >>> 1.0/pytime_statement('sum(range(10))')
 (Statements/sec, does not include any function call overhead).

"""

import sys

__version__ = '1.0.1'
__all__ = ['pytime', 'pytime_statement']

if sys.platform == "win32":
  from time import clock
else:
  from time import time as clock

# Public domain.
def pytime(f, args=(), kwargs={}, Tmax=2.0):
  """
  Calls f many times to determine the average time to execute f.

  Tmax is the maximum time to spend in pytime(), in seconds.
  """
  count = 1
  while True:
    start = clock()
    if args == () and kwargs == {}:
      for i in xrange(count):
        f()
    elif kwargs == {}:
      for i in xrange(count):
        f(*args)
    else:
      for i in xrange(count):
        f(*args, **kwargs)
    T = clock() - start
    if T >= Tmax/4.0: break
    count *= 2
  return T / count


def pytime_statement(stmt, global_dict=None, Tmax=2.0,
                    repeat_count=128):
  """
  Determine time to execute statement (or block) of Python code.

  Here global_dict is the globals dict used for exec, Tmax is the max
  time to spend in pytime_statement(), in sec, and repeat_count is the
  number of times to paste stmt into the inner timing loop (this is
  automatically set to 1 if stmt takes too long).
  """
  if global_dict is None:
    global_dict = globals()

  ns = {}
  code = 'def timed_func():' + ('\n' +
         '\n'.join(['  '+x for x in stmt.split('\n')]))
  exec code in global_dict, ns

  start = clock()
  ns['timed_func']()
  T = clock() - start

  if T >= Tmax/4.0:
    return T
  elif T >= Tmax/4.0/repeat_count:
    return pytime(ns['timed_func'], (), {}, Tmax-T)
  else:
    code = 'def timed_func():' + ('\n' +
           '\n'.join(['  '+x for x in stmt.split('\n')]))*repeat_count
    exec code in global_dict, ns
    return pytime(ns['timed_func'], (), {}, Tmax-T) / repeat_count
