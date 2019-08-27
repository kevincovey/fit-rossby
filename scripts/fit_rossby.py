# Originally written by Stephanie T. Douglas (2012-2014)
# Modified by Kevin Covey (2019)
# under the MIT License (see LICENSE.txt for full details)

import numpy as np
import emcee
import matplotlib.pyplot as plt


def quantile(x,quantiles):
    """ Calculates quantiles - taken from DFM's triangle.py """
    xsorted = sorted(x)
    qvalues = [xsorted[int(q * len(xsorted))] for q in quantiles]
    return list(zip(quantiles,qvalues))

def rossby_model(p,Ro):
    """
    computes the saturated/unsaturated activity model for a given parameter set

    For Ro < turnover, the model values are equal to the saturation level 
    For Ro >= turnover, the model values follow a power-law with slope beta

    Input
    -----
    p : array-like (3)
        parameters for the model: saturation level (expressed as Log L_{whatever}/L_{bol}, turnover_Ro, beta

    Ro : array-like 
        Rossby number values. The model L_{whatever}/L_{bol} values will 
        be computed for these Rossby numbers

    Output
    ------
    y : numpy.ndarray (same size as Ro)
        Model Log L_{whatever}/L_{bol} values corresponding to input Ro

    """

    sat_level,turnover,beta = p[0],p[1],p[2]
    y = np.ones(len(Ro))*sat_level

    un_sat = np.where(Ro>=turnover)[0]
    y[un_sat] = sat_level*(Ro[un_sat]/turnover)**beta

    logy = np.log10(y)

    return logy

    #return y

def lnprob(p,rossby_no,lha_lbol,err_ll): 
    """ 
    Calculates the natural log of the posterior probability for a given model

    The probability calculation uses chi-squared.

    Input
    -----
    p : array-like (3)
        parameters for the model: saturation level, turnover, beta

    rossby_no : array-like 
        Data Rossby number values

    lha_lbol : array-like 
        Data activity values (L_{whatever}/L_{bol} - in my case 
        I was using L_{Halpha}/L_{bol})

    error_ll : array-like
        Uncertainties in the data activity values.

    Output
    ------
    lnprob : float
       natural log of the posterior probability of p given the data

    """
    
    sat_level,turnover,beta,lnf = p[0],p[1],p[2],p[3]
    if ((sat_level>1e-1) or (sat_level<1e-8) or (turnover<0.001)
        or (turnover>2) or (beta>2) or (beta<-6)):
        return -np.inf

    model_ll = rossby_model(p,rossby_no)

    inv_sigma2 = 1.0/(err_ll**2 + model_ll**2*np.exp(2*lnf))
    ln_prob = -0.5*(np.sum((lha_lbol-model_ll)**2*inv_sigma2 - np.log(inv_sigma2)))

    #ln_prob = -0.5*(np.sum((lha_lbol-model_ll)**2/(err_ll**2)))
    #experiment with fitting for the variance, instead of assuming errors
    #ln_prob = -0.5*(np.sum((np.log10(lha_lbol)-np.log10(model_ll))))   #/(variance**2)))
    return ln_prob





def run_rossby_fit(start_p,data_rossby,data_ll,data_ull,
    nwalkers=256,nsteps=10000):
    """
    Sets up the emcee ensemble sampler, runs it, prints out the results,
    then returns the samples.

    Input
    -----
    start_p : (3)
        starting guesses for the three model parameters
        saturation level, turnover point, and power-law slope (beta)

    data_rossby : array-like (ndata)
        Data Rossby number values

    data_ll : array-like (ndata)
        Data activity values (L_{whatever}/L_{bol} - in my case 
        I was using L_{Halpha}/L_{bol})

    data_ull : array-like (ndata)
        Uncertainties in the data activity values.

    Output
    ------
    samples : array-like (nwalkers*nsteps,3)
        all the samples from all the emcee walkers, reshaped so there's
        just one column per parameter

    """
    ndim = 4
    p0 = np.zeros((nwalkers,ndim))

    # initialize the walkers in a tiny gaussian ball around the starting point
    for i in range(nwalkers):
        p0[i] = start_p + (1e-1*np.random.randn(ndim)*start_p)

    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,
        args=[data_rossby,np.log10(data_ll),data_ull])
    pos,prob,state=sampler.run_mcmc(p0,nsteps/2)
    sampler.reset()
    pos,prob,state=sampler.run_mcmc(pos,nsteps)

    sl_mcmc = quantile(sampler.flatchain[:,0],[.16,.5,.84])

    #sl_mcmc.info()
    #print(sl_mcmc)
    to_mcmc = quantile(sampler.flatchain[:,1],[.16,.5,.84])
    #print(to_mcmc)
    be_mcmc = quantile(sampler.flatchain[:,2],[.16,.5,.84])
    #print(be_mcmc)
    var_mcmc = quantile(sampler.flatchain[:,3],[.16,.5,.84])

    
    print('sat_level={0:.7f} +{1:.7f}/-{2:.7f}'.format(
        sl_mcmc[1][1],sl_mcmc[1][1]-sl_mcmc[0][1],sl_mcmc[2][1]-sl_mcmc[1][1]))
    print('turnover={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        to_mcmc[1][1],to_mcmc[1][1]-to_mcmc[0][1],to_mcmc[2][1]-to_mcmc[1][1]))
    print('beta={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        be_mcmc[1][1],be_mcmc[1][1]-be_mcmc[0][1],be_mcmc[2][1]-be_mcmc[1][1]))
    print('var={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        var_mcmc[1][1],var_mcmc[1][1]-var_mcmc[0][1],var_mcmc[2][1]-var_mcmc[1][1]))

    samples = sampler.flatchain

    return samples

def plot_rossby(samples,data_rossby,data_ll,data_ull,
    plotfilename=None,ylabel=r'$L_{X}/L_{bol}$'):
    """ 
    Plot fit results with data 

    Input
    -----
    samples : array-like (nwalkers*nsteps,3)
        all the samples from all the emcee walkers, reshaped so there's
        just one column per parameter

    data_rossby : array-like (ndata)
        Data Rossby number values

    data_ll : array-like (ndata)
        Data activity values (L_{whatever}/L_{bol} - in my case 
        I was using L_{Halpha}/L_{bol})

    data_ull : array-like (ndata)
        Uncertainties in the data activity values.

    plotfilename : string (optional; default=None)
        if not None, the plot will be saved using this filename

    """
    
    sl_mcmc = quantile(samples[:,0],[.16,.5,.84])
    to_mcmc = quantile(samples[:,1],[.16,.5,.84])
    be_mcmc = quantile(samples[:,2],[.16,.5,.84])
    var_mcmc = quantile(samples[:,3],[.16,.5,.84])

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Just trying to reduce the number of plotted points...    
    xl = np.append(np.append(0.001,np.arange(0.08,0.15,0.001)),2.0) 
#    xl = np.arange(0.001,2.0,0.005)
    #for p in list(samples[np.random.randint(len(samples), size=100)]):
    #    ax.plot(xl,rossby_model(p,xl),color='LightGrey')

    sat_level = sl_mcmc[1][1]
    turnover = to_mcmc[1][1]
    x = np.asarray([turnover,2.0])
#    x = np.arange(turnover,2.0,0.001)
    #constant = sat_level/(turnover**-1.)
    #ax.plot(x,constant*(x**-1.),'k--',lw=1.5,label=r'$\beta=\ -1$')
    #constant = sat_level/(turnover**-2.1)
    #ax.plot(x,constant*(x**-2.1),'k-.',lw=1.5,label=r'$\beta=\ -2.1$')
    #constant = sat_level/(turnover**-2.7)
    #ax.plot(x,constant*(x**-2.7),'k:',lw=2,label=r'$\beta=\ -2.7$')

    star_color = 'BlueViolet'
    ax.errorbar(data_rossby,data_ll,data_ull,color=star_color,fmt='.',capsize=0,
        ms=4,mec=star_color)
    ax.plot(xl,10.**rossby_model([sl_mcmc[1][1],to_mcmc[1][1],be_mcmc[1][1]],xl),
        'k-',lw=2,label=r'$\beta=\ {0:.1f}$'.format(be_mcmc[1][1]))
    ax.set_ylabel(ylabel,fontsize='xx-large')
    ax.set_xlabel('R$_o$',fontsize='x-large')
    ax.set_xlim(1e-3,2)
    ax.tick_params(labelsize='x-large')
    #ax.set_xticklabels((0.001,0.01,0.1,1))

    handles, labels = ax.get_legend_handles_labels()
    new_handles = np.append(handles[-1],handles[0:-1])
    new_labels = np.append(labels[-1],labels[0:-1])
    ax.legend(new_handles,new_labels,loc=3) #,
#        title=ylabel+r'\ \propto\ R_o^{\beta}$')

    if plotfilename!=None:
        plt.savefig(plotfilename)

        
def print_pdf(cropchain,filename,col_names=["sat_level,turnover,beta"]):
    f = open(filename,"w")
    f.write("# {}".format(col_names[0]))
    for cname in col_names[1:]:
        f.write(",{}".format(cname))
    f.write("\n")
    
    for i,p in enumerate(cropchain):
        #print p
        f.write(str(p[0]))
        for this_p in p[1:]:
            f.write(",{}".format(this_p))
        f.write("\n")

    f.close()
