# Created by Stephanie T. Douglas, 8 December 2014

import numpy as np
import cPickle
import triangle
import matplotlib.pyplot as plt

from fit_rossby import *
import get_data

# Set up 3 arrays:
# rossby number, L_{X}/L_{bol}, and the associated uncertainty
#data_rossby = 
#data_ll = 
#data_ull = 

pdat,pobs,pobsnr,pobsr = get_data.get_data('P')
hdat,hobs,hobsnr,hobsr = get_data.get_data('H')

pros = pdat.field('ROSSBY')
hros = hdat.field('ROSSBY')
peqw,pueqw = pdat.field('AVG_EQW'),pdat.field('AVG_EQW_ERR')
pll,pull = pdat.field('AVG_LHA'),pdat.field('AVG_LHA_ERR')
heqw,hueqw = hdat.field('AVG_EQW'),hdat.field('AVG_EQW_ERR')
hll,hull = hdat.field('AVG_LHA'),hdat.field('AVG_LHA_ERR')
pbin = (pdat.field('BINARY')>0)
hbin = (hdat.field('BINARY')>0)
pmass = pdat.field('KH_MASS')
hmass = hdat.field('KH_MASS')
pperiods = pdat.field('PERIOD')
hperiods = hdat.field('PERIOD')

ppmem = pdat.field('ADAMPMEM')
hpmem = hdat.field('ROESER_PMEM')
pmem_threshold=70.0

pgood = np.where((pmass<=1.3) & (pmass>0.1) & (pbin==False) & (peqw-pueqw>0)
        & ((ppmem>=pmem_threshold) | (ppmem<0)) & (pperiods>0))[0]
hgood = np.where((hmass<=1.3) & (hmass>0.1) & (hbin==False)  & (heqw-hueqw>0)
        & ((hpmem>=pmem_threshold) | (hpmem<0)) & (hperiods>0))[0]

data_rossby = 10**np.append(pros[pgood],hros[hgood])
sort_order = np.argsort(data_rossby)
data_rossby = data_rossby[sort_order]
#print data_rossby
data_ll = np.append(pll[pgood],hll[hgood])
data_ll = data_ll[sort_order]
#print data_ll
data_ull = np.append(pull[pgood],hull[hgood])
data_ull = data_ull[sort_order]
#print data_ull

# Decide on your starting parameters
start_p = np.asarray([1e-4,0.1,-1.0])

# run the emcee wrapper function
samples = run_rossby_fit(start_p,data_rossby,data_ll,data_ull)

# Plot the output
plot_rossby(samples,data_rossby,data_ll,data_ull)

# Make a triangle plot 
# (this won't be any good for publication - .ps/.eps won't do transparency)
triangle.corner(samples,labels=['sat_level (x10^-4)','turnover','beta'],quantiles=[0.16,0.50,0.84])
# adjust the ticklabels for easier reading
ax = plt.subplot(337)
xticks = ax.get_xticks()
new_labels = []
for xt in xticks:
    new_labels =np.append(new_labels,str(xt*10000))
ax.set_xticklabels(new_labels)

# Uncomment to save the plot
plt.savefig('fit_rossby_corner_temp.png')

# Save the sample positions to a pkl file to be accessed later
outfile = open('fit_rossby_samples_temp.pkl','wb')
cPickle.dump(samples,outfile)
outfile.close()

# Write to a text file for posting online :)
print_pdf(samples,"fit_rossby_samples_temp.csv")
