import numpy as np
import get_data, emcee, triangle, cPickle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from emcee_plot import emcee_plot



def rossby_model(p,Ro):
    sat_level,turnover,beta = p[0],p[1],p[2]
    y = np.ones(len(Ro))*sat_level

    constant = sat_level/(turnover**beta)
    un_sat = np.where(Ro>=turnover)[0]
    y[un_sat] = constant*(Ro[un_sat]**beta)

    return y


def lnprob(p,rossby_no,lha_lbol,err_ll):    
    sat_level,turnover,beta = p[0],p[1],p[2]
    if ((sat_level>1e-1) or (sat_level<1e-8) or (turnover<0.001)
        or (turnover>2) or (beta>2) or (beta<-6)):
        return -np.inf

    model_ll = rossby_model(p,rossby_no)
#    plt.plot(rossby_no,model_ll,'r-',alpha=0.1)
    ln_prob = -0.5*(np.sum((lha_lbol-model_ll)**2/(err_ll**2)))
#    print ln_prob
    return ln_prob


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

#popt,pcov = curve_fit(rossby_model,rossby_no,lha_lbol,
#    p0=np.asarray([1e-5,0.13,-1.0]),sigma=err_ll)
#print popt
#print pcov


ndim=3
nwalkers=100
nsteps=2500
p0 = np.zeros((nwalkers,ndim))
start_p = np.asarray([1e-4,0.1,-1.0])
for i in range(nwalkers):
    p0[i] = start_p + (1e-2*np.random.randn(ndim)*start_p)

sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,
    args=[data_rossby,data_ll,data_ull])
pos,prob,state=sampler.run_mcmc(p0,nsteps/10)
sampler.reset()
plt.close()
pos,prob,state=sampler.run_mcmc(pos,nsteps)


def quantile(x,quantiles):
    xsorted = sorted(x)
    qvalues = [xsorted[int(q * len(xsorted))] for q in quantiles]
    return zip(quantiles,qvalues)

sl_mcmc = quantile(sampler.flatchain[:,0],[.16,.5,.84])
to_mcmc = quantile(sampler.flatchain[:,1],[.16,.5,.84])
be_mcmc = quantile(sampler.flatchain[:,2],[.16,.5,.84])

print 'sat_level={0:.7f} +{1:.7f}/-{2:.7f}'.format(
    sl_mcmc[1][1],sl_mcmc[1][1]-sl_mcmc[0][1],sl_mcmc[2][1]-sl_mcmc[1][1])
print 'turnover={0:.3f} +{1:.3f}/-{2:.3f}'.format(
    to_mcmc[1][1],to_mcmc[1][1]-to_mcmc[0][1],to_mcmc[2][1]-to_mcmc[1][1])
print 'beta={0:.3f} +{1:.3f}/-{2:.3f}'.format(
    be_mcmc[1][1],be_mcmc[1][1]-be_mcmc[0][1],be_mcmc[2][1]-be_mcmc[1][1])




plt.figure()
ax = plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
samples = sampler.chain.reshape((-1,ndim))
xl = np.arange(0.001,2.0,0.005)
for p in samples[np.random.randint(len(samples), size=200)]:
    ax.plot(xl,rossby_model(p,xl),color='LightGrey')

sat_level = sl_mcmc[1][1]
turnover = to_mcmc[1][1]
#ax.plot((0.001,turnover),(sat_level,sat_level),'k-',lw=2)
x = np.arange(turnover,2.0,0.001)
constant = sat_level/(turnover**-1.)
ax.plot(x,constant*(x**-1.),'k--',lw=1.5,label=r'$\beta=\ -1$')
constant = sat_level/(turnover**-2.1)
ax.plot(x,constant*(x**-2.1),'k-.',lw=1.5,label=r'$\beta=\ -2.1$')
constant = sat_level/(turnover**-2.7)
ax.plot(x,constant*(x**-2.7),'k:',lw=2,label=r'$\beta=\ -2.7$')


star_color = 'BlueViolet'
ax.errorbar(data_rossby,data_ll,data_ull,color=star_color,fmt='*',capsize=0,
    ms=12,mec=star_color)
ax.plot(xl,rossby_model([sl_mcmc[1][1],to_mcmc[1][1],be_mcmc[1][1]],xl),
    'k-',lw=2,label=r'$\beta=\ {0:.1f}$'.format(be_mcmc[1][1]))
ax.set_ylabel(r'$L_{H\alpha}/L_{bol}$',fontsize='xx-large')
ax.set_xlabel('Ro',fontsize='x-large')
ax.set_xlim(1e-3,2)
ax.tick_params(labelsize='x-large')
ax.set_xticklabels((0.001,0.01,0.1,1))

handles, labels = ax.get_legend_handles_labels()
new_handles = np.append(handles[-1],handles[0:-1])
new_labels = np.append(labels[-1],labels[0:-1])
ax.legend(new_handles,new_labels,loc=3,
    title=r'$L_{H\alpha}/L_{bol}\ \propto\ Ro^{\beta}$')

plt.savefig('fit_rossby.png')
plt.savefig('fit_rossby.ps')


#emcee_plot(sampler.chain,labels=['sat_level','turnover','beta'])

triangle.corner(sampler.chain.reshape((-1,ndim)),labels=['sat_level (x10^-4)','turnover','beta'],quantiles=[0.16,0.50,0.84])

ax = plt.subplot(337)
xticks = ax.get_xticks()
new_labels = []
for xt in xticks:
    new_labels =np.append(new_labels,str(xt*10000))
ax.set_xticklabels(new_labels)
plt.savefig('fit_rossby_corner.png')
plt.savefig('fit_rossby_corner.ps')

outfile = open('fit_rossby.pkl','wb')
cPickle.dump(samples,outfile)
outfile.close()
