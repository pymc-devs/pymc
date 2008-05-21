from salmon import *

# Create a salmon instance for each datafile.
chum = salmon('chum')
sockeye = salmon('sockeye')
pink = salmon('pink')    

# Fit and make plots.
close('all')

for species in [chum, sockeye, pink]:
    
    observe(species.M,
            species.C,
            obs_mesh = species.abundance, 
            obs_vals = species.frye, 
            obs_V = .25*species.frye)
    
    species.plot()
    
    path = "../../Docs/figs/MMK" + species.name + "reg.pdf"
    # savefig(path)

# show()