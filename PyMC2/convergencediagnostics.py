# Convergence diagnostics

def geweke_zscores(x, first=.1, last=.5, intervals=20):
    """Return z-scores for convergence diagnostics.
	
	Compare the mean of the first % of series with the mean of the last % of 
	series. x is divided into a number of segments for which this difference is
	computed. 

	:Parameters:
	  `x` : series of data,
	  `first` : first fraction of series,
	  `last` : last fraction of series to compare with first,
	  `intervals` : number of segments. 
	  
	:Note: Geweke (1992)
	  """
    # Filter out invalid intervals
    if first + last >= 1:
        raise "Invalid intervals for Geweke convergence analysis",(first,last)
        
    # Initialize list of z-scores
    zscores = []
    
    # Last index value
    end = len(trace) - 1
    
    # Calculate starting indices
    sindices = arange(0, end/2, step = int((end / 2) / intervals))
    
    # Loop over start indices
    for start in sindices:
        
        # Calculate slices
        first_slice = x[start : start + int(first * (end - start))]
        last_slice = x[int(end - last * (end - start)):]
        
        z = (first_slice.mean() - last_slice.mean())
        z /= sqrt(first_slice.std()**2 + last_slice.std()**2)
        
        zscores.append([start, z])
	
	if intervals == None:
	    return zscores[0]    
    else:
    	return zscores
    

def gelman_rubin(x):
# x contains multiple chains
# Transform positive or [0,1] variables using a logarithmic/logittranformation.
# 
