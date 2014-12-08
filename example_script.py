# Created by Stephanie T. Douglas, 8 December 2014

import numpy as np
import cPickle
import triangle
import matplotlib.pyplot as plt
#from emcee_plot import emcee_plot

from fit_rossby import *


# Set up 3 arrays:
# rossby number, L_{X}/L_{bol}, and the associated uncertainty
data_rossby = 
data_ll = 
data_ull = 

# Decide on your starting parameters
start_p = np.asarray([1e-4,0.1,-1.0])

# run the emcee wrapper function
samples = run_rossby_fit(start_p,data_rossby,data_ll,data_ull)

# Plot the output
plot_rossby(samples,data_rossby,data_ll,data_ull)

# Make a triangle plot 
# (this won't be any good for publication - .ps/.eps can't take transparency)
triangle.corner(samples,labels=['sat_level (x10^-4)','turnover','beta'],quantiles=[0.16,0.50,0.84])
# adjust the ticklabels for easier reading
ax = plt.subplot(337)
xticks = ax.get_xticks()
new_labels = []
for xt in xticks:
    new_labels =np.append(new_labels,str(xt*10000))
ax.set_xticklabels(new_labels)

# Uncomment to save the plot
#plt.savefig('fit_rossby_corner.png')



# Save the sampler functions to a pkl file to be accessed later
# (could also write to a text file)
outfile = open('fit_rossby.pkl','wb')
cPickle.dump(samples,outfile)
outfile.close()
