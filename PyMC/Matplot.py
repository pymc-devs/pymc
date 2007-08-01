"""
Plotting module using matplotlib.
"""

# Import matplotlib functions
import matplotlib, pdb
from pylab import bar, hist, plot, xlabel, ylabel, xlim, ylim, savefig, figure, subplot, gca, scatter, setp

# Import numpy functions
from numpy import arange, log, ravel, rank, swapaxes


class PlotFactory(object):
    
    def __init__(self, format='png', backend='TkAgg'):
        # Class initialization
        
        # Specify pylab backend
        matplotlib.use(backend)
        
        # Store output format
        self.format = format
        
        # Store fontmap
        self.fontmap = {1:10, 2:8, 3:6, 4:5, 5:4}
        
        self.plotcount = 0
    
    def plot(self, data, name, color='b', suffix='', new=True, last=True, rows=1, num=1, xlim=()):
        # Plots summary of parameter/node trace
        
        # Find minimum and maximum values for x-axis of density plot
        if not len(xlim):
            alldata = ravel(data)
            xlim = min(alldata), max(alldata)
            del alldata
        
        # If there is only one data array, go ahead and plot it ...
        if rank(data)==1:
            
            print 'Plotting', name
            
            # If new plot, generate new frame
            if new:
                
                figure(self.plotcount, figsize=(10, 6))
                
                self.plotcount += 1
            
            # Call trace
            self.trace(data, name, rows=rows, columns=2, num=num, last=last, color=color)
            
            # Call density
            self.density(data, 'Value', rows=rows, columns=2, num=num+1, last=last, xlim=xlim, color=color)
            
            if last:
                savefig("%s%s.%s" % (name, suffix, self.format))
                #close()
        
        else:
            
            # ... otherwise plot recursively
            tdata = swapaxes(data, 0, 1)
            
            # How many rows?
            _rows = min(4, len(tdata))
            
            for i in range(len(tdata)):
                
                # New plot or adding to existing?
                _new = not i % _rows
                # Current subplot number
                _num = i % _rows * 2 + 1
                # Final subplot of current figure?
                _last = not (_num + 1) % (_rows * 2) or i == (len(tdata) - 1)
                
                self.plot(tdata[i], name+'_'+str(i), color=color, suffix=suffix, new=_new, last=_last, rows=_rows, num=_num, xlim=xlim)

    def density(self, data, name, color='b', suffix='', rows=1, columns=1, num=1, last=True, xlim=()):
        # Internal density plotting specification for handling nested arrays
        try:
            
            # Stand-alone plot or subplot?
            standalone = rows==1 and columns==1 and num==1
            if standalone:
                print 'Generating density of', name
                figure()
            
            # Generate subplot frame
            ax = subplot(rows, columns, num)
            
            # Specify number of bins
            nbins = int(4 + 1.5*log(len(data)))   
            # Generate histogram
            n, bins = matplotlib.mlab.hist(data, nbins)
            # Plot density of histogram
            plot(bins, n, color)
            
            xmin, xmax = xlim
            ax.set_xlim(xmin, xmax)
            
            # Plot options
            if last:
                xlabel(name, fontsize='x-small')
            
            ylabel("Frequency", fontsize='x-small')
            
            # Smaller tick labels
            tlabels = gca().get_xticklabels()
            setp(tlabels, 'fontsize', self.fontmap[rows])
            tlabels = gca().get_yticklabels()
            setp(tlabels, 'fontsize', self.fontmap[rows])
            
            if standalone:
                # Save to file
                savefig("%s%s.%s" % (name, suffix, self.format))
                #close()
        
        except Exception:
            print '... cannot generate desnity'
    
    def histogram(self, data, name, bins=[], nbins=None, suffix='', rows=1, columns=1, num=1, last=True):
        # Internal histogram specification for handling nested arrays
        try:
            
            # Stand-alone plot or subplot?
            standalone = rows==1 and columns==1 and num==1
            if standalone:
                print 'Generating histogram of', name
                figure()
            
            ax = subplot(rows, columns, num)
            
            #Specify number of bins (10 as default)
            if len(bins):
                
                # Generate histogram
                hist(data.tolist(), bins)
                    
            else:
            
                nbins = nbins or int(4 + 1.5*log(len(data)))
                    
                # Generate histogram
                hist(data.tolist(), nbins)
            
            # Plot options
            if last:
                xlabel(name, fontsize='x-small')
            
            ylabel("Frequency", fontsize='x-small')
            
            # Smaller tick labels
            tlabels = gca().get_xticklabels()
            setp(tlabels, 'fontsize', self.fontmap[rows])
            tlabels = gca().get_yticklabels()
            setp(tlabels, 'fontsize', self.fontmap[rows])
            
            if standalone:
                # Save to file
                savefig("%s%s.%s" % (name, suffix, self.format))
                #close()
        
        except Exception:
            print '... cannot generate histogram'

    
    def trace(self, data, name, color='b', suffix='', rows=1, columns=1, num=1, last=True):
        # Internal plotting specification for handling nested arrays
        
        try:
        
            # Stand-alone plot or subplot?
            standalone = rows==1 and columns==1 and num==1
        
            if standalone:
                print 'Plotting', name
                figure()
        
            subplot(rows, columns, num)
            plot(data.tolist(), color)
        
            # Plot options
            if last:
                xlabel('Iteration', fontsize='x-small')
            ylabel(name, fontsize='x-small')
        
            # Smaller tick labels
            tlabels = gca().get_xticklabels()
            setp(tlabels, 'fontsize', self.fontmap[rows])
        
            tlabels = gca().get_yticklabels()
            setp(tlabels, 'fontsize', self.fontmap[rows])
        
            if standalone:
                # Save to file
                savefig("%s%s.%s" % (name, suffix, self.format))
                #close()
                
        except Exception:
            print '... cannot generate trace'
    
    def geweke_plot(self, data, name, color='b', suffix='-diagnostic'):
        # Generate Geweke (1992) diagnostic plots
        
        print 'Plotting', name+suffix
        
        try:
            # Generate new scatter plot
            figure()
            x, y = data
            scatter(x.tolist(), y.tolist())
        
            # Plot options
            xlabel('First iteration', fontsize='x-small')
            ylabel('Z-score', fontsize='x-small')
        
            # Plot lines at +/- 2 sd from zero
            plot((min(x), max(x)), (2, 2), color+'--')
            plot((min(x), max(x)), (-2, -2), color+'--')
        
            # Set plot bound
            ylim(min(-2.5, min(y)), max(2.5, max(y)))
            xlim(0, max(x))
        
            # Save to file
            savefig("%s%s.%s" % (name, suffix, self.format))
            #close()
            
        except Exception:
            print '... cannot generate Geweke plot'
    
    def gof_plot(self, data, name, color='b', suffix='-gof'):
        # Generate goodness-of-fit scatter plot
        
        print 'Plotting', name+suffix
        
        try:
            # Generate new scatter plot
            figure()
            x, y = data
            scatter(x, y)
        
            # Plot x=y line
            lo = min(ravel(data))
            hi = max(ravel(data))
            datarange = hi-lo
            lo -= 0.1*datarange
            hi += 0.1*datarange
            plot((lo, hi), (lo, hi), color)
        
            # Plot options
            xlabel('Observed deviates', fontsize='x-small')
            ylabel('Simulated deviates', fontsize='x-small')
        
            # Save to file
            savefig("%s%s.%s" % (name, suffix, self.format))
            #close()
            
        except Exception:
            print '... cannot generate GOF plot'
    
    def bar_series_plot(self, values, ylab='Y', color='b', suffix=''):
        """Generate bar plot of a series, usually autocorrelation
        or autocovariance."""
        
        try:
            # Extract names
            names = values.keys()
            names.sort()
        
            # Number of plots per page
            rows = min(len(values), 4)
        
            for i,name in enumerate(names):
                print 'Plotting', name+suffix
            
                if not i % rows:
                     # Generate new figure
                    figure(figsize=(10, 6))
            
                # New subplot
                subplot(rows, 1, i - (rows*(i/rows)) + 1)
                y = values[name]
                x = arange(len(y))
                bar(x, y, color=color)
            
                # Set axis bounds
                ylim(-1.0, 1.0)
                xlim(0, len(y))
            
                # Plot options
                ylabel(ylab, fontsize='x-small')
                tlabels = gca().get_yticklabels()
                setp(tlabels, 'fontsize', self.fontmap[rows])
                tlabels = gca().get_xticklabels()
                setp(tlabels, 'fontsize', self.fontmap[rows])
            
                # Save to file
                if not (i+1) % rows or i == len(values)-1:
                
                    # Label X-axis on last subplot
                    xlabel('Lag', fontsize='x-small')
                
                    savefig("%s%s.%s" % (name, suffix, self.format))
                    
        except Exception:
            print '... cannot generate bar series plot'