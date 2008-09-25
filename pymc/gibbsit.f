************************************************************************
*                                                                      *
*  gibbsit                                                    1-10-95  *
*                                                                      *
*    Gibbsit -- Version 2.0, by Adrian E. Raftery and Steven M. Lewis  *
*               (with thanks to Jill Kirby and Jan de Leeuw)           *
*                                                                      *
*  This program calculates the number of iterations required in a MCMC *
*  run.  The user has to specify the precision required.  The program  *
*  returns the number of iterations required to estimate the posterior *
*  cdf of the q-quantile of the quantity of interest (a function of    *
*  the parameters) to within +-r with probability s.  It also gives    *
*  the number of "burn-in" iterations required for the conditional     *
*  distribution given any starting point (of the derived two-state     *
*  process) to be within epsilon of the actual equilibrium distri-     *
*  bution.                                                             *
*                                                                      *
*                                                                      *
*  References:                                                         *
*                                                                      *
*  Raftery, A.E. and Lewis, S.M. (1992).  How many iterations in the   *
*  Gibbs sampler?  In Bayesian Statistics, Vol. 4 (J.M. Bernardo, J.O. *
*  Berger, A.P. Dawid and A.F.M. Smith, eds.). Oxford, U.K.: Oxford    *
*  University Press, 763-773.                                          *
*  This paper is available via the World Wide Web by linking to URL    *
*    http://www.stat.washington.edu/tech.reports/pub/tech.reports      *
*  and then selecting the "How Many Iterations in the Gibbs Sampler"   *
*  link.                                                               *
*  This paper is also available via regular ftp using the following    *
*  commands:                                                           *
*    ftp ftp.stat.washington.edu (or 128.95.17.34)                     *
*    login as anonymous                                                *
*    enter your email address as your password                         *
*    ftp> cd /pub/tech.reports                                         *
*    ftp> get raftery-lewis.ps                                         *
*    ftp> quit                                                         *
*                                                                      *
*  Raftery, A.E. and Lewis, S.M. (1992).  One long run with diagnos-   *
*  tics: Implementation strategies for Markov chain Monte Carlo.       *
*  Statistical Science, Vol. 7, 493-497.                               *
*                                                                      *
*  Raftery, A.E. and Lewis, S.M. (1995).  The number of iterations,    *
*  convergence diagnostics and generic Metropolis algorithms.  In      *
*  Practical Markov Chain Monte Carlo (W.R. Gilks, D.J. Spiegelhalter  *
*  and S. Richardson, eds.). London, U.K.: Chapman and Hall.           *
*  This paper is available via the World Wide Web by linking to URL    *
*    http://www.stat.washington.edu/tech.reports/pub/tech.reports      *
*  and then selecting the "The Number of Iterations, Convergence       *
*  Diagnostics and Generic Metropolis Algorithms" link.                *
*  This paper is also available via regular ftp using the following    *
*  commands:                                                           *
*    ftp ftp.stat.washington.edu (or 128.95.17.34)                     *
*    login as anonymous                                                *
*    enter your email address as your password                         *
*    ftp> cd /pub/tech.reports                                         *
*    ftp> get raftery-lewis2.ps                                        *
*    ftp> quit                                                         *
*                                                                      *
************************************************************************
 
************************************************************************
*                                                                      *
*  Input:                                                              *
*                                                                      *
*  Example values of q, r, s:                                          *
*     0.025, 0.005,  0.95 (for a long-tailed distribution)             *
*     0.025, 0.0125, 0.95 (for a short-tailed distribution);           *
*     0.5, 0.05, 0.95;  0.975, 0.005, 0.95;  etc.                      *
*                                                                      *
*  The result is quite sensitive to r, being proportional to the       *
*  inverse of r^2.                                                     *
*                                                                      *
*  For epsilon, we have always used 0.001.  It seems that the result   *
*  is fairly insensitive to a change of even an order of magnitude in  *
*  epsilon.                                                            *
*                                                                      *
*  One way to use the program is to run it for several specifications  *
*  of r, s and epsilon and vectors q on the same data set.  When one   *
*  is sure that the distribution is fairly short-tailed, such as when  *
*  q=0.025, then r=0.0125 seems sufficient.  However, if one is not    *
*  prepared to assume this, safety seems to require a smaller value of *
*  r, such as 0.005.                                                   *
*                                                                      *
*  The program takes as input the name of a file containing an initial *
*  run from a MCMC sampler.  If the MCMC iterates are independent,     *
*  then the minimum number required to achieve the specified accuracy  *
*  is about $\Phi^{-1} (\frac{1}{2}(1+s))^2 q(1-q)/r^2$ and this would *
*  be a reasonable number to run first.                                *
*  When q=0.025, r=0.005 and s=0.95, this number is 3,748;             *
*  when q=0.025, r=0.0125 and s=0.95, it is 600.                       *
*                                                                      *
*                                                                      *
*  Output:                                                             *
*                                                                      *
*       nmin  = minimum number of iterations, assuming independence,   *
*               required to achieve the specified accuracy.            *
*       nburn = number of iterations to be discarded at the beginning. *
*       nprec = number of subsequent iterations required.              *
*       kthin = skip parameter for a first-order Markov chain.  The    *
*               desired accuracy will be achieved if at most every     *
*               kthin-th iterate is used.                              *
*       kind  = skip parameter sufficient to achieve an independence   *
*               chain.                                                 *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  The development of this computer program was supported by the       *
*  Office of Naval Research under contracts N-00014-88-K-0265 and      *
*  N-00014-91-J-1074 and by the National Institutes of Health under    *
*  grant 5R01HD26330.                                                  *
*                                                                      *
************************************************************************
 
************************************************************************
*                                                                      *
*  Development history:                                                *
*                                                                      *
*  Version 0.3 (May 1, 1991):  Initial version of the program.         *
*  Version 0.4 (June 8, 1991):  This incorporates Jill Kirby's         *
*  suggestion that lots of things be initialized to zero and her code  *
*  that does that.                                                     *
*                                                                      *
*  Version 1.0 (Sept. 14, 1994):  A reorganization of the program by   *
*  Steven Lewis.  This included multiplying nprec by an additional     *
*  missing factor, (2-alpha-beta), modifying the way in which ties are *
*  handled in the calculation of the empirical quantile, and taking    *
*  the logarithm of abs(1-alpha-beta) in the calculation of nburn.     *
*                                                                      *
*  Version 2.0 (Dec. 15, 1994):  Add ability to read a matrix of input *
*  values, add the I_rl statistic to the output, add calculation of    *
*  kind (the spacing needed to achieve an independence chain) to the   *
*  output, add ability to read in a vector of quantiles and add the    *
*  ability to estimate a probability such as described in the Raftery  *
*  and Lewis Bayesian Statistics paper.                                *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  The dimensions are currently set for input series of length at most *
*  50000, but this is easily changed by editing the line below where   *
*  maxiterates is set.                                                 *
*                                                                      *
*  The maximum number of variables which can be read from the input    *
*  file is given by the maxseries parameter.  This is currently set to *
*  20, but is also easily changed.                                     *
*                                                                      *
************************************************************************

      program   gibbsit

      integer   maxiterates
      parameter (maxiterates=50000)

      integer   maxseries
      parameter (maxseries=20)

      integer   maxqcnt
      parameter (maxqcnt=20)

      integer   wasize
      parameter (wasize=maxiterates*2)
 
************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine.  This includes any do-loop counters and similar such    *
*  temporary subscripts and indices.                                   *
*                                                                      *
*    data     - contains the original input MCMC series                *
*    qs,r,s   - parameters used to specify the required precision:     *
*               the q-quantile is to be estimated to within +-r with   *
*               probability s                                          *
*    irl      - the I statistic from the Raftery and Lewis Statistical *
*               Science paper                                          *
*                                                                      *
*    inpfile  - name of file containing the initial MCMC input matrix  *
*                                                                      *
*    workarea - vector used internally by the gibbsmain subroutine     *
*    iteracnt - the number of iterations (ie., rows) in the input data *
*    varcnt   - number of variables (ie., columns) in the input data   *
*    nburn    - the number of iterations required for burn-in          *
*    nprec    - the number of iterations required to achieve the       *
*               specified precision                                    *
*    kthin    - skip parameter for a first-order Markov chain          *
*    kmind    - minimum skip parameter to get an independence chain    *
*    nmin     - minimum number of independent iterates required        *
*    kind     - skip parameter sufficient to achieve an independence   *
*               chain (ie., the greater of ceiling(irl) and kmind)     *
*    ccnt     - number of control parameters entered by the user       *
*    qcnt     - number of quantiles entered by the user                *
*                                                                      *
************************************************************************

      double precision data(maxiterates,maxseries)
      double precision qs(maxqcnt)
      double precision irl

      double precision controlparms(3)
      double precision r
      equivalence (controlparms(1),r)
      double precision s
      equivalence (controlparms(2),s)
      double precision epsilon
      equivalence (controlparms(3),epsilon)

      character inpfile*24

      integer   workarea(wasize)
      integer   argcount
      integer   iteracnt
      integer   varcnt
      integer   nburn
      integer   nprec
      integer   kthin
      integer   kmind
      integer   iargc
      integer   nmin
      integer   kind
      integer   ccnt
      integer   qcnt
      integer   q1
      integer   v1
      integer   rc
 
************************************************************************
*                                                                      *
*  Process whatever command line arguments were passed to gibbsit.     *
*  Currently at most one argument is expected.  If present, the first  *
*  argument contains the name of the input data file.  If not present, *
*  get the name of the input data file from the user's console.        *
*                                                                      *
************************************************************************

      argcount = iargc()
      if (argcount.ge.1) then
        call getarg(1,inpfile)
      else
        write (0,*) 'Enter the name of the input file'
        read (5,*) inpfile
      end if

************************************************************************
*                                                                      *
*  Open the input data file.  If unable to do so, stop the program.    *
*                                                                      *
************************************************************************

      open (unit=7,file=inpfile,status='old',iostat=rc)
      if (rc.ne.0) go to 900

************************************************************************
*                                                                      *
*  Read the input data into the data matrix.                           *
*                                                                      *
************************************************************************

      call matinput(7,maxiterates,maxseries,data,iteracnt,varcnt,rc)
      if (rc.ne.0) then
        write (0,*) 'matinput exited with a nonzero error code of', rc
        go to 900
      end if
 
************************************************************************
*                                                                      *
*  The main program loop follows.  To begin each loop, get the input   *
*  control parameters from the user's console.  If the user enters     *
*  an end-of-data or r=99, stop this program.                          *
*                                                                      *
************************************************************************

  300 write (0,*)
     +  'Enter r,s,epsilon (e.g. .0125 .95 .001).  r=99 to stop'
      call vecinput(5,3,controlparms,ccnt,rc)
      if (rc.gt.0) then
        write (0,*) 'vecinput exited with a nonzero error code of', rc
        go to 900
      end if
      if (rc.lt.0 .or. r.eq.99) go to 900
      if (ccnt.ne.3) then
        write (0,*) 'r, s, and epsilon are all required'
        go to 300
      end if

************************************************************************
*                                                                      *
*  Next get a vector of quantiles at which MCMC settings for each      *
*  variable input series are to be calculated.                         *
*                                                                      *
************************************************************************

      write (0,'(''Enter a vector of quantiles (e.g. .025 .975).  '',
     +  ''q=0 to estimate probability'')')
      call vecinput(5,maxqcnt,qs,qcnt,rc)
      if (rc.ne.0) then
        write (0,*) 'vecinput exited with a nonzero error code of', rc
        go to 900
      end if
 
************************************************************************
*                                                                      *
*  Loop through the vector of quantiles, calculating nmin, kthin,      *
*  nburn, nprec and kind for each variable input series at each given  *
*  quantile value, separating each set by an extra blank line.         *
*                                                                      *
************************************************************************

        do 700 q1=1,qcnt
        write (6,'(/''q = '',f5.3,'', r = '',f6.4,'', s = '',f4.2,
     +    '', epsilon = '',f6.4,'':'')') qs(q1), r, s, epsilon

************************************************************************
*                                                                      *
*  Now execute the gibbmain subroutine once for each variable input    *
*  series for the current quantile value, to perform all of the real   *
*  work.                                                               *
*                                                                      *
************************************************************************

          do 500 v1=1,varcnt
          call gibbmain(data(1,v1),iteracnt,qs(q1),r,s,epsilon,workarea,
     +      nmin,kthin,nburn,nprec,kmind,rc)
          if (rc.ne.0) then
            if (rc.eq.12) then
              write (0,'(''When q=0 the input series must consist of '',
     +          ''only 0''''s and 1''''s'')')
            else
              write (0,'(''gibbmain exited with a nonzero error code '',
     +          ''of '',i2)') rc
            end if
            go to 900
          end if

          irl = dble(nburn + nprec) / dble(nmin)
          kind = max( int(irl + 1.0d0), kmind )
          write (6,'('' ('',i2,'')  kthin='',i3,'', nburn='',i5,
     +      '', nprec='',i8,'', nmin='',i5,'', I='',f6.2,'', kind='',
     +      i3)') v1, kthin, nburn, nprec, nmin, irl, kind
  500     continue

  700   continue

************************************************************************
*                                                                      *
*  Repeat the main program loop with possibly new control parameters   *
*  and/or a new vector of quantiles, leaving an extra blank line at    *
*  the end of the current set of outputs.                              *
*                                                                      *
************************************************************************

      write (6,'()')
      go to 300

************************************************************************
*                                                                      *
*  At this point we are done with the gibbsit program.                 *
*                                                                      *
************************************************************************

  900 stop
      end
 
************************************************************************
*                                                                      *
*  matinput                                                  11-04-94  *
*                                                                      *
*  This subroutine inputs a matrix of double precision numbers from a  *
*  designated file.  The input file is assumed to have the same number *
*  of blank delimited numbers on each record of the file.  There is an *
*  upper limit of twenty numbers per line.  This subroutine reads the  *
*  numbers from the first line of the input file into the first row of *
*  the matrix, from the second line of the input file into the second  *
*  row, and so on.  It is important to check the return code from this *
*  subroutine since there are a number of different error conditions   *
*  which may occur.  Error codes less than 0 are just warnings, so in  *
*  most cases may be ignored.                                          *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  Inputs:                                                             *
*                                                                      *
*    uid      = an integer containing the unit identifier number of an *
*               already opened external input file.  The uid must be a *
*               non-negative number.  If not, matinput will return to  *
*               the caller with an error code of 4 without having read *
*               anything into the output matrix, matout.               *
*                                                                      *
*    rowmax   = an integer containing the allocated number of rows in  *
*               the output matrix, matout.  If the input file contains *
*               more than rowmax records, the first rowmax records     *
*               will be read into the output matrix and matinput will  *
*               return to the caller with an error return code of -4.  *
*               The actual number of rows used is returned in the      *
*               rowused argument.                                      *
*                                                                      *
*    colmax   = an integer containing the allocated number of columns  *
*               in the output matrix, matout.  The maximum number of   *
*               numbers per line which matinput can read into the      *
*               output matrix is the lesser of colmax and twenty.      *
*               The actual number of columns used is returned in the   *
*               colused argument.                                      *
*                                                                      *
*                                                                      *
*  Outputs:                                                            *
*                                                                      *
*    matout   = a double precision matrix in which this subroutine is  *
*               to return the matrix of numbers read in from the input *
*               file.  This matrix consists of rowmax rows by colmax   *
*               columns.                                               *
*                                                                      *
*    rowused  = an integer containing the actual number of rows of the *
*               output matrix, matout, into which matinput has read    *
*               data.  The rest of the output matrix has not been      *
*               altered by this subroutine.                            *
*                                                                      *
*    colused  = an integer containing the actual number of columns of  *
*               the output matrix, matout, into which matinput has     *
*               read data.  The rest of the output matrix has not been *
*               altered by this subroutine.                            *
*                                                                      *
************************************************************************
 
************************************************************************
*                                                                      *
*  Outputs (continued):                                                *
*                                                                      *
*    r15      = an integer valued error return code.  This variable    *
*               is set to 0 if no errors were encountered.             *
*               Otherwise, r15 can assume the following values:        *
*                                                                      *
*                 -4 = the end of file from the input file had not yet *
*                      been reached before running out of the maximum  *
*                      number of rows available in the output matrix.  *
*                      The first rowmax records from the input file    *
*                      will have been read into the output matrix.     *
*                  4 = the input file unit identifier was negative.    *
*                  8 = the rowmax argument was not a postitive number. *
*                 12 = the oneparse subroutine returned with a nonzero *
*                      error return code.                              *
*                 16 = an error occurred while converting one of the   *
*                      input numbers into internal double precision    *
*                      format.  Any input numbers after the number     *
*                      causing the error will not have been read into  *
*                      output matrix.                                  *
*               No other possible values are currently in use.         *
*                                                                      *
************************************************************************

      subroutine matinput(uid,rowmax,colmax,matout,rowused,colused,r15)

      integer   uid
      integer   rowmax
      integer   colmax
      double precision matout(rowmax,colmax)
      integer   rowused
      integer   colused
      integer   r15

************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine such as loop counters, indices and so forth.             *
*                                                                      *
*    curterms - parsed version of the current input record             *
*    currecrd - the most recently read record from the input file      *
*    delimit  - a single character to be used as the separator between *
*               numbers in the input file                              *
*    collimit - lesser of colmax and twenty                            *
*    curcnt   - number of tokens parsed from the current input record  *
*                                                                      *
************************************************************************

      character curterms(20)*24
      character currecrd*512
      character delimit*1 /' '/

      integer   collimit
      integer   curcnt
      integer   c1
      integer   rc
 
************************************************************************
*                                                                      *
*  Make sure that the external unit identifier of the input file is a  *
*  non-negative number.  If it isn't, return to the caller with an     *
*  error code of 4.                                                    *
*                                                                      *
************************************************************************

      if (uid.lt.0) then
        write (0,*) 'unit identifier passed to matinput is negative'
        r15 = 4
        return
      end if

************************************************************************
*                                                                      *
*  Make sure that the rowmax argument is greater than 0.  If it isn't, *
*  return to the caller with an error code of 8.                       *
*                                                                      *
************************************************************************

      if (rowmax.le.0) then
        write (0,*) 'output matrix must have a positive number of rows'
        r15 = 8
        return
      end if

************************************************************************
*                                                                      *
*  The maximum number of columns of the output matrix which matinput   *
*  can read into is the lesser of the colmax argument and twenty, but  *
*  it must be at least 1.                                              *
*                                                                      *
*  For the moment, initialize the number of columns used to this just  *
*  determined maximum number of columns.                               *
*                                                                      *
************************************************************************

      if (colmax.lt.20) then
        collimit = max(colmax,1)
      else
        collimit = 20
      end if
      colused = collimit

************************************************************************
*                                                                      *
*  Initialize the number of rows used argument to 0.                   *
*                                                                      *
************************************************************************

      rowused = 0
 
************************************************************************
*                                                                      *
*  Read the next record from the input file as a character string.  If *
*  there are no more records in the input file, we have completed our  *
*  job successfully.  In this latter case we can jump to the end of    *
*  the subroutine to set the error code to 0 before returning.         *
*                                                                      *
*  Note that I chose to read at least one non-blank record from the    *
*  input file before checking the number of rows used in the output    *
*  matrix so that the error code will be set to 0 in the case where    *
*  the number of non-blank records in the input file is precisely      *
*  equal to the number of rows in the output matrix.                   *
*                                                                      *
************************************************************************

  200 read (uid,'(a)',end=600) currecrd

************************************************************************
*                                                                      *
*  Call the oneparse subroutine to parse the current record into up to *
*  collimit blank separated tokens.  If any blank lines are read from  *
*  the input file, these should be ignored.                            *
*                                                                      *
************************************************************************

      call oneparse(currecrd,delimit,collimit,curterms,curcnt,rc)
      if (rc.ne.0) then
        write (0,*) 'oneparse exited with a nonzero error code of', rc
        r15 = 12
        return
      end if
      if (curcnt.lt.1) go to 200

************************************************************************
*                                                                      *
*  Make sure that the number of rows used in the output matrix is      *
*  still less than the maximum number of rows in the matrix.  If it    *
*  is not, then we have read as much as we can into the matrix.  We    *
*  can return to the caller but set the error code to -4 to warn the   *
*  caller that not all the data from the input file could be read      *
*  into the output matrix.                                             *
*                                                                      *
************************************************************************

      if (rowused.ge.rowmax) then
        r15 = -4
        return
      end if

************************************************************************
*                                                                      *
*  To make sure that every significant element of the output matrix    *
*  will have been set by the matinput subroutine, this subroutine sets *
*  the actual number of columns used equal to the fewest number of     *
*  tokens read in from any one record of the input file.               *
*                                                                      *
************************************************************************

      if (curcnt.lt.colused) colused = curcnt
 
************************************************************************
*                                                                      *
*  Increment the number of rows used in the output matrix.             *
*                                                                      *
************************************************************************

      rowused = rowused + 1

************************************************************************
*                                                                      *
*  After parsing the input record into separate numbers, convert each  *
*  of the input quantities into internal double precision numbers.     *
*                                                                      *
************************************************************************

        do 300 c1=1,colused
        read (curterms(c1),'(f24.0)',err=400) matout(rowused,c1)
  300   continue

************************************************************************
*                                                                      *
*  Go read the next record from the input file.                        *
*                                                                      *
************************************************************************

      go to 200

************************************************************************
*                                                                      *
*  An error occurred in trying to convert one of the input numbers     *
*  into a floating point number.  The most prudent thing to do would   *
*  be to return to the caller without finishing reading from the input *
*  file.  The error return code is set to 16 before returning.         *
*                                                                      *
************************************************************************

  400 r15 = 16
      return

************************************************************************
*                                                                      *
*  Everything has gone as expected if we made it to this point in the  *
*  program, so return to the caller with the good news.                *
*                                                                      *
************************************************************************

  600 r15 = 0
      return
      end
 
************************************************************************
*                                                                      *
*  vecinput                                                  12-07-94  *
*                                                                      *
*  This subroutine reads a vector of double precision numbers from a   *
*  designated file (which may be the standard input).  Only a single   *
*  line is read from the input file.  This line is assumed to consist  *
*  of up to twenty blank delimited numbers.                            *
*                                                                      *
*  Vecinput is a trimmed down version of the matinput subroutine.      *
*  Instead of reading in an entire matrix, vecinput only reads in a    *
*  single line of input into a vector.  Since matrix input need not be *
*  supported by this subroutine, the code may be greatly simplified.   *
*  First, there is no need in vecinput to keep track of row numbers in *
*  a matrix, so any code referring to row number may be removed.  The  *
*  other key simplification possible in this subroutine is a result of *
*  the fact that only a single line of input is to be read by this     *
*  subroutine, whereas the matinput subroutine always attempts to read *
*  more than one line of input.  This difference in behavior is not    *
*  worth worrying about, except in the case where the input is being   *
*  read from the user's terminal.  Any extra attempted reads to the    *
*  terminal would not ordinarily be expected by a user and hence need  *
*  to be prevented.  It is particularly this last reason which led me  *
*  to implement a separate subroutine to read in a simple vector of    *
*  numbers (which is very likely to be entered interactively).         *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  Inputs:                                                             *
*                                                                      *
*    uid      = an integer containing the unit identifier number of an *
*               already opened external input file.  The uid must be a *
*               non-negative number.  If not, vecinput will return to  *
*               the caller with an error code of 4 without having read *
*               anything into the output vector, vecout.               *
*                                                                      *
*    vecmax   = an integer containing the allocated number of elements *
*               in the output vector, vecout.  The maximum number of   *
*               numbers which vecinput can read into the output vector *
*               is the lesser of vecmax and twenty.  The actual number *
*               of elements used is returned in the vecused argument.  *
*                                                                      *
************************************************************************
 
************************************************************************
*                                                                      *
*  Outputs:                                                            *
*                                                                      *
*    vecout   = a double precision vector in which this subroutine is  *
*               to return the vector of numbers read from the single   *
*               line of the input file.  This vector must contain at   *
*               least vecmax elements.                                 *
*                                                                      *
*    vecused  = an integer containing the actual number of elements of *
*               the output vector, vecout, into which vecinput has     *
*               read data.  The rest of the output vector will not     *
*               have been modified by this subroutine.                 *
*                                                                      *
*    r15      = an integer valued error return code.  This variable    *
*               is set to 0 if no errors were encountered.             *
*               Otherwise, r15 can assume the following values:        *
*                                                                      *
*                 -4 = the user entered an end-of-data when prompted   *
*                      for the single line of input.                   *
*                  4 = the input file unit identifier was negative.    *
*                  8 = the oneparse subroutine returned with a nonzero *
*                      error return code.                              *
*                 12 = an error occurred while converting one of the   *
*                      input numbers into internal double precision    *
*                      format.  Any input numbers after the number     *
*                      causing the error will not have been read into  *
*                      output vector.                                  *
*               No other possible values are currently in use.         *
*                                                                      *
************************************************************************

      subroutine vecinput(uid,vecmax,vecout,vecused,r15)

      integer   uid
      integer   vecmax
      double precision vecout(vecmax)
      integer   vecused
      integer   r15
 
************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine such as loop counters, indices and so forth.             *
*                                                                      *
*    septerms - parsed version of the line read from the input file    *
*    charinpt - character form of the line read from the input file    *
*    delimit  - a single character to be used as the separator between *
*               numbers in the input file                              *
*    veclimit - lesser of vecmax and twenty                            *
*                                                                      *
************************************************************************

      character septerms(20)*24
      character charinpt*512
      character delimit*1 /' '/

      integer   veclimit
      integer   v1
      integer   rc

************************************************************************
*                                                                      *
*  Make sure that the external unit identifier of the input file is a  *
*  non-negative number.  If it isn't, return to the caller with an     *
*  error code of 4.                                                    *
*                                                                      *
************************************************************************

      if (uid.lt.0) then
        write (0,*) 'unit identifier passed to vecinput is negative'
        r15 = 4
        return
      end if

************************************************************************
*                                                                      *
*  The maximum number of elements of the output vector which vecinput  *
*  can read into is the lesser of the vecmax argument and twenty, but  *
*  it must be at least 1.                                              *
*                                                                      *
************************************************************************

      if (vecmax.lt.20) then
        veclimit = max(vecmax,1)
      else
        veclimit = 20
      end if
 
************************************************************************
*                                                                      *
*  Read a single line from the input file as a character string.  If   *
*  an end-of-data or an end-of-file occurs, we just need to set the    *
*  error return code to -4 before returning to the caller.             *
*                                                                      *
************************************************************************

      read (uid,'(a)',end=400) charinpt

************************************************************************
*                                                                      *
*  Call the oneparse subroutine to parse the line read from the input  *
*  file into up to veclimit blank separated terms.                     *
*                                                                      *
************************************************************************

      call oneparse(charinpt,delimit,veclimit,septerms,vecused,rc)
      if (rc.ne.0) then
        write (0,*) 'oneparse exited with a nonzero error code of', rc
        r15 = 8
        return
      end if

************************************************************************
*                                                                      *
*  After parsing the input record into separate numbers, convert each  *
*  of the input quantities into internal double precision numbers.     *
*                                                                      *
************************************************************************

        do 300 v1=1,vecused
        read (septerms(v1),'(f24.0)',err=500) vecout(v1)
  300   continue

************************************************************************
*                                                                      *
*  Everything has gone as expected if we made it to this point in the  *
*  program, so return to the caller with the good news.                *
*                                                                      *
************************************************************************

      r15 = 0
      return
 
************************************************************************
*                                                                      *
*  An end-of-data or an end-of-file occurred when we attempted to read *
*  the single line from the input file.  Set the error return code to  *
*  -4 before returning to the caller.                                  *
*                                                                      *
************************************************************************

  400 r15 = -4
      return

************************************************************************
*                                                                      *
*  An error occurred in trying to convert one of the input numbers     *
*  into a floating point number.  Set the error return code to 12      *
*  before returning to the caller.                                     *
*                                                                      *
************************************************************************

  500 r15 = 12
      return
      end
 
************************************************************************
*                                                                      *
*  oneparse                                                   2-17-94  *
*                                                                      *
*  This subroutine is a very rudimentary parser.  The input character  *
*  string passed as the first argument is parsed into 0 or more tokens *
*  where the tokens are separated by instances of the single character *
*  delimiter passed as the delimit argument.  The separate tokens are  *
*  returned in the output character vector, tokens.                    *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  Inputs:                                                             *
*                                                                      *
*    instring = a character string to be parsed into separate tokens.  *
*                                                                      *
*    delimit  = a single character to be used to separate the input    *
*               into individual tokens.                                *
*                                                                      *
*    maxtok   = an integer containing the maximum number of tokens     *
*               which may be returned.  This is the number of elements *
*               available in the tokens vector.                        *
*                                                                      *
*                                                                      *
*  Outputs:                                                            *
*                                                                      *
*    tokens   = a vector of character strings containing the parsed    *
*               version of the input character string, instring.  Each *
*               entry in tokens will be left justified (leading blanks *
*               removed).  The number of tokens found in the input     *
*               string is returned in tokcnt.                          *
*                                                                      *
*    tokcnt   = an integer containing the number of tokens actually    *
*               returned in tokens.  This will be a number between 0   *
*               and maxtok, inclusive.                                 *
*                                                                      *
*    r15      = an integer valued error return code.  This variable    *
*               is set to 0 if no errors were encountered.             *
*               Otherwise, r15 can assume the following values:        *
*                                                                      *
*                  4 = the input character string contained more than  *
*                      maxtok tokens.  The first maxtok tokens have    *
*                      been returned in tokens and tokcnt has been set *
*                      equal to maxtok.                                *
*               No other possible values are currently in use.         *
*                                                                      *
************************************************************************

      subroutine oneparse(instring,delimit,maxtok,tokens,tokcnt,r15)

      character instring*(*)
      character delimit*1
      integer   maxtok
      character tokens(maxtok)*(*)
      integer   tokcnt
      integer   r15
 
************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine such as loop counters, indices and so forth.             *
*                                                                      *
************************************************************************

      integer   inlen
      integer   index
      integer   bpos
      integer   dpos
      integer   epos
      integer   len
      integer   tn
      integer   pn

************************************************************************
*                                                                      *
*  Parse the input character string into its separate tokens.  The     *
*  input string is assumed to contain from 0 to maxtok tokens          *
*  separated by the token separator character, delimit.                *
*                                                                      *
************************************************************************

      inlen = len(instring)
      bpos = 1
      tn = 0

************************************************************************
*                                                                      *
*  Locate the beginning and the end of the next token within the       *
*  input string.                                                       *
*                                                                      *
*  First, find the next nonblank character in the input string.        *
*                                                                      *
************************************************************************

  400   do 450 pn=bpos,inlen
        if (instring(pn:pn).ne.' ') go to 500
  450   continue

************************************************************************
*                                                                      *
*  If no nonblank characters were found in the input string, there     *
*  are no more tokens in the input, so we have finished parsing this   *
*  input string.                                                       *
*                                                                      *
*  At this point set tokcnt to the number of tokens found and set the  *
*  error return code to indicate that all went as would be expected    *
*  before returning to the caller.                                     *
*                                                                      *
************************************************************************

      tokcnt = tn
      r15 = 0
      return
 
************************************************************************
*                                                                      *
*  The beginning of the next token has been located.  Increment the    *
*  subscript to use for this token.  The element of the tokens vector  *
*  addressed by this subscript will be the one used to return the      *
*  current token.                                                      *
*                                                                      *
*  If this subscript is greater than maxtok, the passed input string   *
*  contained more tokens than can be returned in the tokens vector.    *
*  In this case, tokcnt is set equal to maxtok and the error return    *
*  code is set to 4 before returning to the caller.                    *
*                                                                      *
*  To find the end of this token, find the next token separator or the *
*  end of the input string if there are no more separators.            *
*                                                                      *
************************************************************************

  500 tn = tn + 1
      if (tn.gt.maxtok) then
        tokcnt = maxtok
        r15 = 4
        return
      end if

      bpos = pn
      dpos = index(instring(bpos:),delimit)
      if (dpos.eq.0) dpos = inlen
      epos = bpos + dpos - 2

************************************************************************
*                                                                      *
*  We now have the positions of both the beginning and the end of the  *
*  next token within the input string, so save this token in the       *
*  correct element of the tokens vector.                               *
*                                                                      *
************************************************************************

      tokens(tn) = instring(bpos:epos)

************************************************************************
*                                                                      *
*  Start looking for the next token immediately following the most     *
*  recently located token separator (the one which terminated the      *
*  previous token).                                                    *
*                                                                      *
************************************************************************

      bpos = epos + 2
      go to 400

      end
 
************************************************************************
*                                                                      *
*  gibbmain                                                   1-09-95  *
*                                                                      *
*  This program calculates the number of iterations required in a run  *
*  of MCMC.  The user has to specify the precision required.  This     *
*  subroutine returns the number of iterations required to estimate    *
*  the posterior cdf of the q-quantile of the quantity of interest (a  *
*  function of the parameters) to within +-r with probability s.  It   *
*  also gives the number of "burn-in" iterations required for the      *
*  conditional distribution given any starting point (of the derived   *
*  two-state process) to be within epsilon of the actual equilibrium   *
*  distribution.                                                       *
*                                                                      *
*  If q<=0, then gibbmain is to treat the original input series as a   *
*  vector of 0-1 outcome variables.  In this case no quantile needs to *
*  be found.  Instead, this subroutine just needs to calculate kthin,  *
*  nburn, nprec and kmind tuning parameters such that a MCMC run based *
*  on these tuning parameters should be adequate for estimating the    *
*  probability of an outcome of 1 within the prescribed (by r, s, and  *
*  epsilon) probability.                                               *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  Inputs:                                                             *
*                                                                      *
*    original = a double precision vector containing the original MCMC *
*               generated series of parameter estimates.  This vector  *
*               contains iteracnt elements.                            *
*                                                                      *
*    iteracnt = an integer containing the number of actual iterations  *
*               provided in the sample MCMC output series, original.   *
*                                                                      *
*    q,r,s    = double precision numbers in which the caller specifies *
*               the required precision:  the q-quantile is to be       *
*               estimated to within r with probability s.              *
*                                                                      *
*    epsilon  = a double precision number containing the half width of *
*               the tolerance interval required for the q-quantile.    *
*                                                                      *
*    work     = an integer vector passed to various subroutines to     *
*               hold a number of internal vectors.  There must be at   *
*               least (iteracnt * 2) elements in this vector.          *
*                                                                      *
************************************************************************
 
************************************************************************
*                                                                      *
*  Outputs:                                                            *
*                                                                      *
*    nmin     = an integer which will be set to the minimum number of  *
*               independent Gibbs iterates required to achieve the     *
*               specified accuracy for the q-quantile.                 *
*                                                                      *
*    kthin    = an integer which will be set to the skip parameter     *
*               sufficient to produce a first-order Markov chain.      *
*                                                                      *
*    nburn    = an integer which will be set to the number of          *
*               iterations to be discarded at the beginning of the     *
*               simulation, i.e. the number of burn-in iterations.     *
*                                                                      *
*    nprec    = an integer which will be set to the number of          *
*               iterations not including the burn-in iterations which  *
*               need to be obtained in order to attain the precision   *
*               specified by the values of the q, r and s input        *
*               parameters.                                            *
*                                                                      *
*    kmind    = an integer which will be set to the minimum skip       *
*               parameter sufficient to produce an independence chain. *
*                                                                      *
*    r15      = an integer valued error return code.  This variable    *
*               is set to 0 if no errors were encountered.             *
*               Otherwise, r15 can assume the following values:        *
*                                                                      *
*                 12 = the original input vector contains something    *
*                      other than a 0 or 1 even though q<=0.           *
*               No other possible values are currently in use.         *
*                                                                      *
************************************************************************
C Modified by D. Huard to wrap using F2Py.

      subroutine gibbmain(original,iteracnt,q,r,s,epsilon,work,nmin,
     +  kthin,nburn,nprec,kmind,r15)

      integer   iteracnt
      double precision original(iteracnt)
      double precision q
      double precision r
      double precision s
      double precision epsilon
      integer   work(iteracnt*2)
      integer   nmin
      integer   kthin
      integer   nburn
      integer   nprec
      integer   kmind
      integer   r15

CF2PY INTEGER, INTENT(HIDE), DEPEND(ORIGINAL) :: ITERACNT = LEN(ORIGINAL)
CF2PY DOUBLE PRECISION DIMENSION(ITERACNT), INTENT(IN) :: ORIGINAL
CF2PY DOUBLE PRECISION INTENT(IN) :: Q,R,S,EPSILON
CF2PY DOUBLE PRECISION INTENT(CACHE,HIDE), DIMENSION(2*ITERACNT) :: WORK
CF2PY INTEGER, INTENT(OUT) :: NMIN, KTHIN, NBURN, NPREC, KMIND
CF2PY INTEGER, INTENT(HIDE) :: R15
************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine.  This includes any do-loop counters and similar such    *
*  temporary subscripts and indices.                                   *
*                                                                      *
*    cutpt    - the q-th empirical quantile                            *
*    qhat     - when q=0, proportion of 1's in the input data vector,  *
*               when q>0, qhat is set equal to the passed value of q   *
*    g2       - G2 for the test of first-order vs second-order Markov  *
*    bic      - the corresponding BIC value                            *
*    phi      - \PHI^{-1} ((s+1)/2)                                    *
*    alpha    - probability of moving from below the cutpt to above    *
*    beta     - probability of moving from above the cutpt to below    *
*    probsum  - sum of alpha + beta                                    *
*                                                                      *
*  The first iteracnt elements of the work vector will be used to      *
*  store a binary 0-1 series indicating which elements are less than   *
*  or equal to the cutpt (set to 1) and which elements are less than   *
*  the cutpt (set to 0).                                               *
*                                                                      *
*  The remaining iteracnt elements of the work vector are to be used   *
*  to hold thinned versions of the 0-1 series, where the amount of     *
*  thinning is determined by the current value of kthin (or kmind).    *
*  That is, for each proposed value of kthin (or kmind), only every    *
*  kthin-th (or kmind-th) element of the 0-1 series is copied to this  *
*  thinned copy of the series.                                         *
*                                                                      *
*  ixkstart is the subscript of the first element of the thinned       *
*  series.  That is, ixkstart = iteracnt + 1.                          *
*                                                                      *
*  thincnt is the current length of the thinned series.                *
*                                                                      *
************************************************************************

      double precision empquant
      double precision cutpt
      double precision qhat
      double precision g2
      double precision bic
      double precision phi

      double precision alpha
      double precision beta
      double precision probsum

      double precision tmp1
      double precision tmp2

      real      ppnd7

      integer   ixkstart
      integer   thincnt
      integer   i1
      integer   rc
 
************************************************************************
*                                                                      *
*  If the q argument is a postive number, interpret it as the quantile *
*  which is to be ascertained using MCMC.  It should be a positive     *
*  number less than 1.  Set qhat to the passed value of q (we will use *
*  qhat later when we calculate nmin).                                 *
*                                                                      *
************************************************************************

      if (q.gt.0.0d0) then
        qhat = q

************************************************************************
*                                                                      *
*  Find the q-th quantile of the original MCMC series of parameter     *
*  estimates.                                                          *
*                                                                      *
************************************************************************

        cutpt = empquant(original,iteracnt,qhat,work)

************************************************************************
*                                                                      *
*  Calculate a binary 0-1 series indicating which elements are less    *
*  than or equal to the cutpt (set to 1) and which elements are        *
*  greater than the cutpt (set to 0).  The resulting series is stored  *
*  in the work vector.                                                 *
*                                                                      *
************************************************************************

        call dichot(original,iteracnt,cutpt,work)

************************************************************************
*                                                                      *
*  Otherwise treat the original input series as a binary 0-1 series of *
*  outcomes whose probability needs to be estimated using MCMC.  This  *
*  is easily accomplished by copying the input series into the first   *
*  iteracnt elements of the work vector, converting the double preci-  *
*  sion input into an equivalent integer vector of 0's and 1's.  For   *
*  this case we will also need to set qhat equal to the proportion of  *
*  1's in the original input data vector.                              *
*                                                                      *
************************************************************************

      else

        qhat = 0.0d0

          do 300 i1=1,iteracnt
          if (original(i1).eq.0.0d0 .or. original(i1).eq.1.0d0) then
            work(i1) = int( original(i1) )
            qhat = qhat + original(i1)
          else
            r15 = 12
            return
          end if
  300     continue

        qhat = qhat / dble( iteracnt )

      end if
 
************************************************************************
*                                                                      *
*  Find kthin, the degree of thinning at which the indicator series is *
*  first-order Markov.                                                 *
*                                                                      *
************************************************************************

      ixkstart = iteracnt + 1
      kthin = 1

  500 call thin(work,iteracnt,kthin,work(ixkstart),thincnt)
      call mctest(work(ixkstart),thincnt,g2,bic)
      if (bic.le.0.0d0) go to 600
      kthin = kthin + 1
      go to 500

************************************************************************
*                                                                      *
*  Calculate both the alpha and beta transition probabilities (in the  *
*  Cox & Miller parametrization) of the two state first-order Markov   *
*  chain determined above.                                             *
*                                                                      *
************************************************************************

  600 call mcest(work(ixkstart),thincnt,alpha,beta)
      kmind = kthin
      go to 750

************************************************************************
*                                                                      *
*  Now compute just how big the spacing needs to be so that a thinned  *
*  chain would no longer be a Markov chain, but rather would be an     *
*  independence chain.  This thinning parameter must be at least as    *
*  large as the thinning parameter required for a first-order Markov   *
*  chain.                                                              *
*                                                                      *
************************************************************************

  700 call thin(work,iteracnt,kmind,work(ixkstart),thincnt)
  750 call indtest(work(ixkstart),thincnt,g2,bic)
      if (bic.le.0.0d0) go to 800
      kmind = kmind + 1
      go to 700
 
************************************************************************
*                                                                      *
*  Estimate the first-order Markov chain parameters and find the       *
*  burn-in and precision number of required iterations.                *
*                                                                      *
************************************************************************

  800 probsum = alpha + beta
      tmp1 = dlog(probsum * epsilon / max(alpha,beta)) /
     +  dlog( dabs(1.0d0 - probsum) )
      nburn = int( tmp1 + 1.0d0 ) * kthin

************************************************************************
*                                                                      *
*  Note:  ppnd7 is the routine that implements AS algorithm 241.       *
*  It calculates the specified percentile of the Normal distribution.  *
*                                                                      *
************************************************************************

      phi = dble(ppnd7( ((real(s) + 1.0) / 2.0), rc ))
      tmp2 = (2.0d0 - probsum) * alpha * beta * phi**2 / (probsum**3 *
     +  r**2)
      nprec = int( tmp2 + 1.0d0 ) * kthin
      nmin = int( ((1.0d0-qhat) * qhat * phi**2 / r**2) + 1.0d0 )

************************************************************************
*                                                                      *
*  At this point we have calculated nmin, kthin, nburn, nprec and      *
*  kmind, so we can return to the calling program.                     *
*                                                                      *
************************************************************************

      r15 = 0
      return
      end
 
************************************************************************
*                                                                      *
*  empquant                                                   9-13-94  *
*                                                                      *
*  This function finds the q-th empirical quantile of the input double *
*  precsion series, data, of length iteracnt.                          *
*                                                                      *
*  The algorithm used by this subroutine is the one used in the SPLUS  *
*  quantile function.                                                  *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  Inputs:                                                             *
*                                                                      *
*    data     = a double precision vector of numbers whose q-th        *
*               empirical quantile is to be calculated.                *
*                                                                      *
*    iteracnt = an integer containing the number of elements in the    *
*               input data vector, data.  There must also be this      *
*               many elements in the work vector.                      *
*                                                                      *
*    q        = a double precision number between 0.0d0 and 1.0d0,     *
*               inclusive, specifying which empirical quantile is      *
*               wanted.                                                *
*                                                                      *
*    work     = a double precision vector to be used as a work area    *
*               for the sort subroutine called by empquant.  This      *
*               vector must contain at least iteracnt elements.        *
*                                                                      *
*                                                                      *
*  Outputs:                                                            *
*                                                                      *
*    empquant = a double precision number corresponding to the q-th    *
*               level of the sorted vector of input values.            *
*                                                                      *
************************************************************************

      function empquant(data,iteracnt,q,work)

      double precision empquant
      integer   iteracnt
      double precision data(iteracnt)
      double precision q
      double precision work(*)

************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine.  This includes any do-loop counters and similar such    *
*  temporary subscripts and indices.                                   *
*                                                                      *
************************************************************************

      double precision order
      double precision fract

      integer   low
      integer   high
      integer   i1
 
************************************************************************
*                                                                      *
*  Copy the input series of double precision numbers into the work     *
*  area provided by the caller.  In this way the original input will   *
*  not be modified by this subroutine.                                 *
*                                                                      *
************************************************************************

        do 300 i1=1,iteracnt
        work(i1) = data(i1)
  300   continue

************************************************************************
*                                                                      *
*  Sort the input series into ascending order.                         *
*                                                                      *
************************************************************************

      call ssort(work,work,iteracnt,int(1))

************************************************************************
*                                                                      *
*  Now locate the q-th empirical quantile.  This apparently longer     *
*  than necessary calculation is used so as to appropriately handle    *
*  the case where there are two or more identical values at the        *
*  requested quantile.                                                 *
*                                                                      *
************************************************************************

      order = dble(iteracnt-1) * q + 1.0d0
      fract = mod(order, 1.0d0)
      low = max(int(order), 1)
      high = min(low+1, iteracnt)
      empquant = (1.0d0 - fract) * work(low) + fract * work(high)

      return
      end
 
************************************************************************
*                                                                      *
*  dichot                                                     9-13-94  *
*                                                                      *
*  This subroutine takes a double precision vector, data, of length    *
*  iteracnt and converts it into a 0-1 series in zt, depending on      *
*  which elements of data are less than or greater than cutpt.         *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  Inputs:                                                             *
*                                                                      *
*    data     = a double precision vector containing a series of       *
*               numbers which are to be compared to cutpt in order to  *
*               determine which elements of zt are to be set to 1 and  *
*               which are to be set to 0.                              *
*                                                                      *
*    iteracnt = an integer containing the number of elements in the    *
*               input data vector.                                     *
*                                                                      *
*    cutpt    = a double precision number indicating the boundary      *
*               about which the input data vector is to be dichoto-    *
*               mized, i.e. set to 1 when less than or equal to the    *
*               cutpoint and to 0 when greater than the cutpoint.      *
*                                                                      *
*                                                                      *
*  Outputs:                                                            *
*                                                                      *
*    zt       = an integer vector containing zeros and ones depending  *
*               on whether or not the corresponding elements of data   *
*               were less than the cutpoint or not.                    *
*                                                                      *
************************************************************************

      subroutine dichot(data,iteracnt,cutpt,zt)

      integer   iteracnt
      double precision data(iteracnt)
      double precision cutpt
      integer   zt(iteracnt)

************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine.  This includes any do-loop counters and similar such    *
*  temporary subscripts and indices.                                   *
*                                                                      *
************************************************************************

      integer   i1
 
************************************************************************
*                                                                      *
*  If the entry in the input data vector is less than or equal to the  *
*  cutpoint, set the corresponding element of zt to 1, otherwise set   *
*  it to 0.                                                            *
*                                                                      *
************************************************************************

        do 500 i1=1,iteracnt
        if (data(i1).le.cutpt) then
          zt(i1) = 1
        else
          zt(i1) = 0
        end if
  500   continue

      return
      end
 
************************************************************************
*                                                                      *
*  thin                                                       9-13-94  *
*                                                                      *
*  This subroutine takes the integer-valued vector series of length    *
*  iteracnt and outputs elements 1,1+kthin,1+2kthin,1+3kthin,... in    *
*  the result vector.                                                  *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  Inputs:                                                             *
*                                                                      *
*    series   = an integer vector containing the sequence of numbers   *
*               from which this subroutine is to select every kthin'th *
*               number to be copied to the output vector, starting     *
*               with the first number.  There are iteracnt elements in *
*               this vector.                                           *
*                                                                      *
*    iteracnt = an integer containing the number of elements in the    *
*               input vector of numbers to be thinned, series.  If     *
*               kthin can be as little as 1, the output result vector  *
*               must also contain iteracnt elements.                   *
*                                                                      *
*    kthin    = an integer specifying the interval between elements of *
*               the input data vector, series, which are to be copied  *
*               to the output vector, result.  If kthin is 1, then all *
*               of series is to be copied to result.  If kthin is 2,   *
*               then only every other element of series is to be       *
*               copied to result.  If kthin is 3, then every third     *
*               element is copied and so forth.                        *
*                                                                      *
*                                                                      *
*  Outputs:                                                            *
*                                                                      *
*    result   = an integer vector containing the thinned subset of the *
*               input data vector, series, starting with the first     *
*               element and copying every kthin'th from there on.  The *
*               number of meaningful elements in this vector will be   *
*               returned as thincnt.                                   *
*                                                                      *
*    thincnt  = an integer containing the number of elements actually  *
*               copied to the result vector.                           *
*                                                                      *
************************************************************************

      subroutine thin(series,iteracnt,kthin,result,thincnt)

      integer   iteracnt
      integer   series(iteracnt)
      integer   kthin
      integer   result(iteracnt)
      integer   thincnt
 
************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine.  This includes any do-loop counters and similar such    *
*  temporary subscripts and indices.                                   *
*                                                                      *
************************************************************************

      integer   from
      integer   i1

************************************************************************
*                                                                      *
*  The specified subset of the input data vector, series, is copied to *
*  sequential elements of the output vector, result.  Stop copying     *
*  when the entries in the input data vector run out.                  *
*                                                                      *
************************************************************************

        do 300 i1=1,iteracnt
        from = (i1-1) * kthin + 1
        if (from.gt.iteracnt) go to 600
        result(i1) = series(from)
  300   continue

************************************************************************
*                                                                      *
*  Calculate how many elements have been copied to the output vector,  *
*  result, and return this number as thincnt.                          *
*                                                                      *
************************************************************************

  600 thincnt = i1 - 1
      return
      end
 
************************************************************************
*                                                                      *
*  mctest                                                    12-05-94  *
*                                                                      *
*  This subroutine tests for a first-order Markov chain against a      *
*  second-order Markov chain using the log-linear modeling             *
*  formulation.  Here the first-order model is the [12][23] model,     *
*  while the 2nd-order model is the saturated model.  The [12][23]     *
*  model has closed form estimates - see Bishop, Feinberg and Holland. *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  Inputs:                                                             *
*                                                                      *
*    data     = an integer vector containing the series of 0's and 1's *
*               for which this subroutine is to determine whether a    *
*               first-order Markov chain is sufficient or whether a    *
*               second-order Markov chain is needed to model the data. *
*               There must be at least datacnt elements in the data    *
*               vector.                                                *
*                                                                      *
*    datacnt  = an integer containing the number of elements in the    *
*               data argument.                                         *
*                                                                      *
*                                                                      *
*  Outputs:                                                            *
*                                                                      *
*    g2       = a double precision number in which this subroutine is  *
*               to return the log likelihood ratio statistic for       *
*               testing a second-order Markov chain against only a     *
*               first-order Markov chain.  Bishop, Feinberg and        *
*               Holland denote this statistic as G2.                   *
*                                                                      *
*    bic      = a double precision number in which this subroutine is  *
*               to return the BIC value corresponding to the log       *
*               likelihood ratio statistic, g2.                        *
*                                                                      *
************************************************************************

      subroutine mctest(data,datacnt,g2,bic)

      integer   datacnt
      integer   data(datacnt)
      double precision g2
      double precision bic

************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine.  This includes any do-loop counters and similar such    *
*  temporary subscripts and indices.                                   *
*                                                                      *
************************************************************************

      double precision fitted
      double precision focus

      integer   tran(2,2,2)
      integer   i1
      integer   i2
      integer   i3
 
************************************************************************
*                                                                      *
*  Initialize the transition counts array to all zeroes.               *
*                                                                      *
************************************************************************

        do 300 i1=1,2
          do 200 i2=1,2
            do 100 i3=1,2
            tran(i1,i2,i3) = 0
  100       continue
  200     continue
  300   continue

************************************************************************
*                                                                      *
*  Count up the number of occurrences of each possible type of         *
*  transition.  Keep these counts in the transition counts array.      *
*                                                                      *
************************************************************************

        do 400 i1=3,datacnt
        tran( data(i1-2)+1, data(i1-1)+1, data(i1)+1 ) =
     +    tran( data(i1-2)+1, data(i1-1)+1, data(i1)+1 ) + 1
  400   continue

************************************************************************
*                                                                      *
*  Compute the log likelihood ratio statistic for testing a second-    *
*  order Markov chain against only a first-order Markov chain.  This   *
*  is Bishop, Feinberg and Holland's G2 statistic.                     *
*                                                                      *
************************************************************************

      g2 = 0.0d0

        do 700 i1=1,2
          do 600 i2=1,2
            do 500 i3=1,2
            if (tran(i1,i2,i3).eq.0) go to 500
            fitted = dble( (tran(i1,i2,1) + tran(i1,i2,2)) *
     +        (tran(1,i2,i3) + tran(2,i2,i3)) ) / dble( tran(1,i2,1) +
     +        tran(1,i2,2) + tran(2,i2,1) + tran(2,i2,2) )
            focus = dble( tran(i1,i2,i3) )
            g2 = g2 + dlog( focus / fitted ) * focus
  500       continue
  600     continue
  700   continue

      g2 = g2 * 2.0d0

************************************************************************
*                                                                      *
*  Finally calculate the associated bic statistic and return to the    *
*  caller.                                                             *
*                                                                      *
************************************************************************

      bic = g2 - dlog( dble(datacnt-2) ) * 2.0d0
      return
      end
 
************************************************************************
*                                                                      *
*  indtest                                                    11-23-94 *
*                                                                      *
*  This subroutine tests for an independence chain against a first-    *
*  order Markov chain using the log-linear modeling formulation.  In   *
*  our case the independence model is the [1][2][3] model, while the   *
*  first-order model is the [12][23] model.  Both the [1][2][3] and    *
*  the [12][23] models have closed form estimates - see Bishop,        *
*  Feinberg and Holland (1975).                                        *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  Inputs:                                                             *
*                                                                      *
*    data     = an integer vector containing the series of 0's and 1's *
*               for which this subroutine is to determine whether an   *
*               independence chain is sufficient or whether a first-   *
*               order Markov chain is needed to model the data.  There *
*               must be at least datacnt elements in the data vector.  *
*                                                                      *
*    datacnt  = an integer containing the number of elements in the    *
*               data argument.                                         *
*                                                                      *
*                                                                      *
*  Outputs:                                                            *
*                                                                      *
*    g2       = a double precision number in which this subroutine is  *
*               to return the log likelihood ratio statistic for       *
*               testing a first-order Markov chain against simply an   *
*               independence chain.  Bishop, Feinberg and Holland      *
*               denote this statistic as G2.                           *
*                                                                      *
*    bic      = a double precision number in which this subroutine is  *
*               to return the BIC value corresponding to the log       *
*               likelihood ratio statistic, g2.                        *
*                                                                      *
************************************************************************

      subroutine indtest(data,datacnt,g2,bic)

      integer   datacnt
      integer   data(datacnt)
      double precision g2
      double precision bic

************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine.  This includes any do-loop counters and similar such    *
*  temporary subscripts and indices.                                   *
*                                                                      *
************************************************************************

      double precision fitted
      double precision focus
      double precision dcm1

      integer   tran(2,2)
      integer   i1
      integer   i2
 
************************************************************************
*                                                                      *
*  Initialize the transition counts array to all zeroes.               *
*                                                                      *
************************************************************************

        do 300 i1=1,2
          do 200 i2=1,2
            tran(i1,i2) = 0
  200     continue
  300   continue

************************************************************************
*                                                                      *
*  Count up the number of occurrences of each possible type of         *
*  transition.  Keep these counts in the transition counts array.      *
*                                                                      *
************************************************************************

        do 400 i1=2,datacnt
        tran( data(i1-1)+1, data(i1)+1 ) = tran( data(i1-1)+1,
     +    data(i1)+1 ) + 1
  400   continue

************************************************************************
*                                                                      *
*  Compute the log likelihood ratio statistic for testing a first-     *
*  order Markov chain against simply an independence chain.  This is   *
*  Bishop, Feinberg and Holland's G2 statistic.                        *
*                                                                      *
************************************************************************

      dcm1 = dble( datacnt-1 )
      g2 = 0.0d0

        do 700 i1=1,2
          do 600 i2=1,2
            if (tran(i1,i2).eq.0) go to 600
            fitted = dble( (tran(i1,1) + tran(i1,2)) * (tran(1,i2) +
     +        tran(2,i2)) ) / dcm1
            focus = dble( tran(i1,i2) )
            g2 = g2 + dlog( focus / fitted ) * focus
  600     continue
  700   continue

      g2 = g2 * 2.0d0

************************************************************************
*                                                                      *
*  Finally calculate the associated bic statistic and return to the    *
*  caller.  Note that the first-order Markov chain model contains just *
*  one more parameter than does the independence chain model, so p=1.  *
*                                                                      *
************************************************************************

      bic = g2 - dlog( dcm1 )
      return
      end
 
************************************************************************
*                                                                      *
*  mcest                                                     12-05-94  *
*                                                                      *
*  Estimate the parameters of a first-order Markov chain (in the Cox   *
*  & Miller parametrization) from a series of binary, i.e. 0-1, data   *
*  passed in the data vector argument.                                 *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  Inputs:                                                             *
*                                                                      *
*    data     = an integer vector containing the series of 0's and 1's *
*               from which this subroutine is to calculate empirical   *
*               probabilities of a transition from a 0 to a 1 or a     *
*               transition from a 1 to a 0.  There must be at least    *
*               datacnt elements in this vector.                       *
*                                                                      *
*    datacnt  = an integer containing the number of elements in the    *
*               data argument.                                         *
*                                                                      *
*                                                                      *
*  Outputs:                                                            *
*                                                                      *
*    alpha    = a double precision number in which this subroutine is  *
*               to return the empirical probability of a 1 following   *
*               a 0 in the input data vector.                          *
*                                                                      *
*    beta     = a double precision number in which this subroutine is  *
*               to return the empirical probability of a 0 following   *
*               a 1 in the input data vector.                          *
*                                                                      *
************************************************************************

      subroutine mcest(data,datacnt,alpha,beta)

      integer   datacnt
      integer   data(datacnt)
      double precision alpha
      double precision beta

************************************************************************
*                                                                      *
*  The following variables hold various temporary values used in this  *
*  subroutine.  This includes any do-loop counters and similar such    *
*  temporary subscripts and indices.                                   *
*                                                                      *
************************************************************************

      integer   tran(2,2)
      integer   i1
      integer   i2
 
************************************************************************
*                                                                      *
*  Initialize the transition counts array to all zeroes.               *
*                                                                      *
************************************************************************

        do 200 i1=1,2
          do 100 i2=1,2
          tran(i1,i2) = 0
  100     continue
  200   continue

************************************************************************
*                                                                      *
*  Count up the number of occurrences of each possible type of         *
*  transition.  Keep these counts in the transition counts array.      *
*                                                                      *
************************************************************************

        do 400 i1=2,datacnt
        tran( data(i1-1)+1, data(i1)+1 ) = tran( data(i1-1)+1,
     +    data(i1)+1 ) + 1
  400   continue

************************************************************************
*                                                                      *
*  Calculate the empirical transition probabilities between 0's and    *
*  1's in the input (returned in alpha) and between 1's and 0's in the *
*  input (returned in beta).                                           *
*                                                                      *
************************************************************************

      alpha = dble(tran(1,2)) / dble( (tran(1,1) + tran(1,2)) )
      beta = dble(tran(2,1)) / dble( (tran(2,1) + tran(2,2)) )

      return
      end
 
        REAL FUNCTION PPND7(P,IFAULT)

*       ALGORITHM AS241 APPL. STATIST. (1988) VOL. 37, NO. 3, 477-
*       484.

*       Produces the normal deviate Z corresponding to a given lower
*       tail area of P; Z is accurate to about 1 part in 10**7.

*       The hash sums below are the sums of the mantissas of the
*       coefficients.   They are included for use in checking
*       transcription.

        REAL ZERO, ONE, HALF, SPLIT1, SPLIT2, CONST1, CONST2, A0, A1,
     +          A2, A3, B1, B2, B3, C0, C1, C2, C3, D1, D2, E0, E1, E2,
     +          E3, F1, F2, P, Q, R
        PARAMETER (ZERO = 0.0, ONE = 1.0, HALF = 0.5,
     +          SPLIT1 = 0.425, SPLIT2 = 5.0,
     +          CONST1 = 0.180625, CONST2 = 1.6)
        INTEGER IFAULT

*       Coefficients for P close to 0.5

        PARAMETER (A0 = 3.3871327179E+00, A1 = 5.0434271938E+01,
     +             A2 = 1.5929113202E+02, A3 = 5.9109374720E+01,
     +             B1 = 1.7895169469E+01, B2 = 7.8757757664E+01,
     +             B3 = 6.7187563600E+01)
*       HASH SUM AB    32.3184577772

*       Coefficients for P not close to 0, 0.5 or 1.

        PARAMETER (C0 = 1.4234372777E+00, C1 = 2.7568153900E+00,
     +             C2 = 1.3067284816E+00, C3 = 1.7023821103E-01,
     +             D1 = 7.3700164250E-01, D2 = 1.2021132975E-01)
*       HASH SUM CD    15.7614929821

*       Coefficients for P near 0 or 1.

        PARAMETER (E0 = 6.6579051150E+00, E1 = 3.0812263860E+00,
     +             E2 = 4.2868294337E-01, E3 = 1.7337203997E-02,
     +             F1 = 2.4197894225E-01, F2 = 1.2258202635E-02)
*        HASH SUM EF    19.4052910204

        IFAULT = 0
        Q = P - HALF
        IF (ABS(Q) .LE. SPLIT1) THEN
          R = CONST1 - Q * Q
          PPND7 = Q * (((A3 * R + A2) * R + A1) * R + A0) /
     +                (((B3 * R + B2) * R + B1) * R + ONE)
          RETURN
        ELSE
          IF (Q .LT. ZERO) THEN
            R = P
          ELSE
            R = ONE - P
          END IF
          IF (R .LE. ZERO) THEN
            IFAULT = 1
            PPND7 = ZERO
            RETURN
          END IF
          R = SQRT(-LOG(R))
          IF (R .LE. SPLIT2) THEN
            R = R - CONST2
            PPND7 = (((C3 * R + C2) * R + C1) * R + C0) /
     +               ((D2 * R + D1) * R + ONE)
          ELSE
            R = R - SPLIT2
            PPND7 = (((E3 * R + E2) * R + E1) * R + E0) /
     +               ((F2 * R + F1) * R + ONE)
          END IF
          IF (Q .LT. ZERO) PPND7 = - PPND7
          RETURN
        END IF
        END
 
      SUBROUTINE SSORT(X,Y,N,KFLAG)
****BEGIN PROLOGUE   SSORT
****REVISION  OCTOBER 1,1980
****CATEGORY NO.  M1
****KEYWORD(S) SORTING,SORT,SINGLETON QUICKSORT,QUICKSORT
****DATE WRITTEN  NOVEMBER,1976
****AUTHOR  JONES R.E., WISNIEWSKI J.A. (SLA)
****PURPOSE
*         SSORT SORTS ARRAY X AND OPTIONALLY MAKES THE SAME
*         INTERCHANGES IN ARRAY Y.  THE ARRAY X MAY BE SORTED IN
*         INCREASING ORDER OR DECREASING ORDER.  A SLIGHTLY MODIFIED
*         QUICKSORT ALGORITHM IS USED.
****DESCRIPTION
*     SANDIA MATHEMATICAL PROGRAM LIBRARY
*     APPLIED MATHEMATICS DIVISION 2646
*     SANDIA LABORATORIES
*     ALBUQUERQUE, NEW MEXICO  87185
*     CONTROL DATA 6600/7600  VERSION 8.1  AUGUST 1980
*
*     WRITTEN BY RONDALL E JONES
*     MODIFIED BY JOHN A. WISNIEWSKI TO USE THE SINGLETON QUICKSORT
*     ALGORITHM. DATE 18 NOVEMBER 1976.
*
*     ABSTRACT
*         SSORT SORTS ARRAY X AND OPTIONALLY MAKES THE SAME
*         INTERCHANGES IN ARRAY Y.  THE ARRAY X MAY BE SORTED IN
*         INCREASING ORDER OR DECREASING ORDER.  A SLIGHTLY MODIFIED
*         QUICKSORT ALGORITHM IS USED.
*
*     REFERENCE
*         SINGLETON,R.C., ALGORITHM 347, AN EFFICIENT ALGORITHM FOR
*         SORTING WITH MINIMAL STORAGE, CACM,12(3),1969,185-7.
*
*     DESCRIPTION OF PARAMETERS
*         X - ARRAY OF VALUES TO BE SORTED (USUALLY ABSCISSAS)
*         Y - ARRAY TO BE (OPTIONALLY) CARRIED ALONG
*         N - NUMBER OF VALUES IN ARRAY X TO BE SORTED
*         KFLAG - CONTROL PARAMETER
*             =2  MEANS SORT X IN INCREASING ORDER AND CARRY Y ALONG.
*             =1  MEANS SORT X IN INCREASING ORDER (IGNORING Y)
*             =-1 MEANS SORT X IN DECREASING ORDER (IGNORING Y)
*             =-2 MEANS SORT X IN DECREASING ORDER AND CARRY Y ALONG.
*
****REFERENCE(S)
*         SINGLETON,R.C., ALGORITHM 347, AN EFFICIENT ALGORITHM FOR
*         SORTING WITH MINIMAL STORAGE, CACM,12(3),1969,185-7.
****END PROLOGUE

      INTEGER   I, IJ, IL(21), IU(21), J, K, KFLAG, KK, L, M, N, NN
      DOUBLE PRECISION  R, T, TT, TTY, TY, X(N), Y(N)

****FIRST EXECUTABLE STATEMENT    SSORT
      NN = N
      KK = IABS(KFLAG)
*
* ALTER ARRAY X TO GET DECREASING ORDER IF NEEDED
*
   15 IF (KFLAG.GE.1) GO TO 30
      DO 20 I=1,NN
   20 X(I) = -X(I)
   30 GO TO (100,200),KK
*
* SORT X ONLY
*
  100 CONTINUE
      M = 1
      I = 1
      J = NN
      R = .375
  110 IF (I .EQ. J) GO TO 155
  115 IF (R .GT. .5898437) GO TO 120
      R = R+3.90625E-2
      GO TO 125
  120 R = R-.21875
  125 K = I
*                                  SELECT A CENTRAL ELEMENT OF THE
*                                  ARRAY AND SAVE IT IN LOCATION T
      IJ = I + IDINT( DBLE(J-I) * R )
      T = X(IJ)
*                                  IF FIRST ELEMENT OF ARRAY IS GREATER
*                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 130
      X(IJ) = X(I)
      X(I) = T
      T = X(IJ)
  130 L = J
*                                  IF LAST ELEMENT OF ARRAY IS LESS THAN
*                                  T, INTERCHANGE WITH T
      IF (X(J) .GE. T) GO TO 140
      X(IJ) = X(J)
      X(J) = T
      T = X(IJ)
*                                  IF FIRST ELEMENT OF ARRAY IS GREATER
*                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 140
      X(IJ) = X(I)
      X(I) = T
      T = X(IJ)
      GO TO 140
  135 TT = X(L)
      X(L) = X(K)
      X(K) = TT
*                                  FIND AN ELEMENT IN THE SECOND HALF OF
*                                  THE ARRAY WHICH IS SMALLER THAN T
  140 L = L-1
      IF (X(L) .GT. T) GO TO 140
*                                  FIND AN ELEMENT IN THE FIRST HALF OF
*                                  THE ARRAY WHICH IS GREATER THAN T
  145 K = K+1
      IF (X(K) .LT. T) GO TO 145
*                                  INTERCHANGE THESE ELEMENTS
      IF (K .LE. L) GO TO 135
*                                  SAVE UPPER AND LOWER SUBSCRIPTS OF
*                                  THE ARRAY YET TO BE SORTED
      IF (L-I .LE. J-K) GO TO 150
      IL(M) = I
      IU(M) = L
      I = K
      M = M+1
      GO TO 160
  150 IL(M) = K
      IU(M) = J
      J = L
      M = M+1
      GO TO 160
*                                  BEGIN AGAIN ON ANOTHER PORTION OF
*                                  THE UNSORTED ARRAY
  155 M = M-1
      IF (M .EQ. 0) GO TO 300
      I = IL(M)
      J = IU(M)
  160 IF (J-I .GE. 1) GO TO 125
      IF (I .EQ. 1) GO TO 110
      I = I-1
  165 I = I+1
      IF (I .EQ. J) GO TO 155
      T = X(I+1)
      IF (X(I) .LE. T) GO TO 165
      K = I
  170 X(K+1) = X(K)
      K = K-1
      IF (T .LT. X(K)) GO TO 170
      X(K+1) = T
      GO TO 165
*
* SORT X AND CARRY Y ALONG
*
  200 CONTINUE
      M = 1
      I = 1
      J = NN
      R = .375
  210 IF (I .EQ. J) GO TO 255
  215 IF (R .GT. .5898437) GO TO 220
      R = R+3.90625E-2
      GO TO 225
  220 R = R-.21875
  225 K = I
*                                  SELECT A CENTRAL ELEMENT OF THE
*                                  ARRAY AND SAVE IT IN LOCATION T
      IJ = I + IDINT( DBLE(J-I) * R )
      T = X(IJ)
      TY = Y(IJ)
*                                  IF FIRST ELEMENT OF ARRAY IS GREATER
*                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 230
      X(IJ) = X(I)
      X(I) = T
      T = X(IJ)
      Y(IJ) = Y(I)
      Y(I) = TY
      TY = Y(IJ)
  230 L = J
*                                  IF LAST ELEMENT OF ARRAY IS LESS THAN
*                                  T, INTERCHANGE WITH T
      IF (X(J) .GE. T) GO TO 240
      X(IJ) = X(J)
      X(J) = T
      T = X(IJ)
      Y(IJ) = Y(J)
      Y(J) = TY
      TY = Y(IJ)
*                                  IF FIRST ELEMENT OF ARRAY IS GREATER
*                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 240
      X(IJ) = X(I)
      X(I) = T
      T = X(IJ)
      Y(IJ) = Y(I)
      Y(I) = TY
      TY = Y(IJ)
      GO TO 240
  235 TT = X(L)
      X(L) = X(K)
      X(K) = TT
      TTY = Y(L)
      Y(L) = Y(K)
      Y(K) = TTY
*                                  FIND AN ELEMENT IN THE SECOND HALF OF
*                                  THE ARRAY WHICH IS SMALLER THAN T
  240 L = L-1
      IF (X(L) .GT. T) GO TO 240
*                                  FIND AN ELEMENT IN THE FIRST HALF OF
*                                  THE ARRAY WHICH IS GREATER THAN T
  245 K = K+1
      IF (X(K) .LT. T) GO TO 245
*                                  INTERCHANGE THESE ELEMENTS
      IF (K .LE. L) GO TO 235
*                                  SAVE UPPER AND LOWER SUBSCRIPTS OF
*                                  THE ARRAY YET TO BE SORTED
      IF (L-I .LE. J-K) GO TO 250
      IL(M) = I
      IU(M) = L
      I = K
      M = M+1
      GO TO 260
  250 IL(M) = K
      IU(M) = J
      J = L
      M = M+1
      GO TO 260
*                                  BEGIN AGAIN ON ANOTHER PORTION OF
*                                  THE UNSORTED ARRAY
  255 M = M-1
      IF (M .EQ. 0) GO TO 300
      I = IL(M)
      J = IU(M)
  260 IF (J-I .GE. 1) GO TO 225
      IF (I .EQ. 1) GO TO 210
      I = I-1
  265 I = I+1
      IF (I .EQ. J) GO TO 255
      T = X(I+1)
      TY = Y(I+1)
      IF (X(I) .LE. T) GO TO 265
      K = I
  270 X(K+1) = X(K)
      Y(K+1) = Y(K)
      K = K-1
      IF (T .LT. X(K)) GO TO 270
      X(K+1) = T
      Y(K+1) = TY
      GO TO 265
*
* CLEAN UP
*
  300 IF (KFLAG.GE.1) RETURN
      DO 310 I=1,NN
  310 X(I) = -X(I)
      RETURN
      END
