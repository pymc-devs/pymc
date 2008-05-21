from getdata import *
from matplotlib.toolkits.basemap import Basemap
from matplotlib import *
from pylab import *


# A function from the scipy cookbook
def cmap_map(function,cmap):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red','green','blue'):         step_dict[key] = map(lambda x: x[0], cdict[key])
    step_list = reduce(lambda x, y: x+y, step_dict.values())
    step_list = array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : array(cmap(step)[0:3])
    old_LUT = array(map( reduced_cmap, step_list))
    new_LUT = array(map( function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i,key in enumerate(('red','green','blue')):
        this_cdict = {}
        for j,step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j,i]
            elif new_LUT[j,i]!=old_LUT[j,i]:
                this_cdict[step] = new_LUT[j,i]
        colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


close('all')

llcrnrlon = 0
urcrnrlon = 30
llcrnrlat = 40
urcrnrlat = lat_hi
m = Basemap(projection='cyl',
            # lat_1 = llcrnrlat*.5,
            # lon_1 = lon_mid,
            # lat_2 = urcrnrlat*1.5,
            # lon_2 = lon_mid,
            llcrnrlon = llcrnrlon, 
            llcrnrlat = llcrnrlat,
            urcrnrlon = urcrnrlon,
            urcrnrlat = urcrnrlat,
            resolution = 'l')

N_plot = 20
lat_plot = linspace(llcrnrlat, urcrnrlat, N_plot)
lon_plot = linspace(llcrnrlon, urcrnrlon, N_plot)

x_plot, y_plot = m(lon_plot, lat_plot)
lon_plot, lat_plot = meshgrid(lon_plot, lat_plot)
x_plot, y_plot = meshgrid(x_plot, y_plot)
pts_plot = zeros((N_plot,N_plot,2),dtype=float)
pts_plot[:,:,0]= lon_plot
pts_plot[:,:,1]= lat_plot

def make_map(fun=None):

    figure()

    m.drawcoastlines()
    m.drawmapboundary()
    # m.fillcontinents()

    # draw parallels and meridians.
    # m.drawparallels(arange(llcrnrlat,urcrnrlat,10.))
    # m.drawmeridians(arange(llcrnrlon,urcrnrlon,10.))

    x, y = m(lons, lats)
    m.plot(x, y,'r.',markersize=8)
    if fun is not None:
        # m.contourf(x_plot,y_plot,fun(pts_plot))
        m.imshow(fun(pts_plot), cmap=cmap_map(lambda x: x*.9+.1, cm.bone))
        colorbar()
        
    m.drawlsmask((255,255,255,0),(5,0,30,255),lakes=True)