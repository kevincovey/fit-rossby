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

def rossby_model_log(parameters,Ro):
    """
    computes the saturated/unsaturated activity model for a given parameter set

    For Ro < turnover, the model values are equal to the saturation level 
    For Ro >= turnover, the model values follow a power-law with slope beta

    Inputs and outputs are in log space (ie, saturation level is -3., rather than 10.**(-3.); similar for loglxlbol values)

    Input
    -----
    parameters : array-like (3)
        parameters for the model: saturation level (expressed as Log L_{whatever}/L_{bol}, turnover_Ro, beta

    Ro : array-like 
        Rossby number values. The model Log L_{whatever}/L_{bol} values will 
        be computed for these Rossby numbers

    Output
    ------
     : numpy.ndarray (same size as Ro)
        Model Log L_{whatever}/L_{bol} values corresponding to input Ro

    """

    #save the parameters with intuitive names
    sat_level,turnover,beta = parameters[0],parameters[1],parameters[2]

    #calculate the pivot constant that ensures that the power law reaches the saturation point at the turnover point
    pivot_constant = sat_level - beta * np.log10(turnover)

    #define the Log_LxLbol array and fill with saturated level datapoints
    Log_LxLbol = np.ones(len(Ro))*sat_level

    #find unsaturated objects and calculate their Log_LxLbols based on the assumed power law behavior
    un_sat = np.where(Ro>=turnover)[0]
    Log_LxLbol[un_sat] = pivot_constant + beta * np.log10(Ro[un_sat])

    return Log_LxLbol



def lnprior(parameters):
    """
    simple method of setting (flat) priors on model parameters

    If input parameters are within the priors, a (constant) likelihood is returned; 
    if the input parameters are outside the priors, a negative infinity is returned
    to indicate an unacceptable fit.

    Input
    -----
    parameters : array-like (3)
        parameters for the model: saturation level (expressed as Log L_{whatever}/L_{bol}, turnover_Ro, beta

    Output
    ------
     : value
        0.0 if parameters are within priors; -np.inf if not.
    """
    
    sat_level, turnover, beta, lnf = parameters[0], parameters[1], parameters[2], parameters[3]
    if -4.0 < sat_level < -2.0 and 0.05 < turnover < 0.3 and -3.5 < beta < -1.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf


def lnlike(parameters, rossby_no, log_LxLbol ,err_ll): 
    """ 
    Calculates the natural log of the likelihood for a given model fit to a given input dataset (with errors).

    Input
    -----
    parameters : array-like (4)
        parameters for the model: saturation level, turnover, beta, multiplicative error inflator

    rossby_no : array-like 
        Data Rossby number values

    log_LxLbol : array-like 
        Data activity values (L_{whatever}/L_{bol} - in the original case, LxLbol

    error_ll : array-like
        Uncertainties in the data activity values.

    Output
    ------
    lnprob : float
       natural log of the likelihood of the model given the data

    """
    
    sat_level, turnover, beta, lnf = parameters[0], parameters[1], parameters[2], parameters[3]
    #if ((sat_level>1e-1) or (sat_level<1e-8) or (turnover<0.001)    ## stephanie's original method of setting priors;
    #    or (turnover>2) or (beta>2) or (beta<-6)):                  ## now offloaded to lnprior
    #    return -np.inf

    model_ll = rossby_model_log(parameters, rossby_no)

    #inv_sigma2 = 1.0/(err_ll**2)  ## inverse sigma assuming only quoted errors
    inv_sigma2 = 1.0/(err_ll**2 + model_ll**2*np.exp(2*lnf))  ## inverse sigma assuming errors are underestimated by some multiplicative factor
    ln_like = -0.5*(np.sum((log_LxLbol-model_ll)**2*inv_sigma2 - np.log(inv_sigma2)))

    return ln_like


def lnprob(parameters, rossby_no, log_LxLbol, err_ll):
    """ 
    Calculates the natural log of the probability of a model, given a set of priors, the defined likelihood function, and the observed data

    Input
    -----
    parameters : array-like (4)
        parameters for the model: saturation level, turnover, beta, multiplicative error inflator

    rossby_no : array-like 
        Data Rossby number values

    log_LxLbol : array-like 
        Data activity values (L_{whatever}/L_{bol} - in the original case, LxLbol

    error_ll : array-like
        Uncertainties in the data activity values.

    Output
    ------
    lnprob : float
       natural log of the likelihood of the model given the data and the priors 
       (by adding prior and model likelihood terms, which are 
        calculated by lnprior() and lnlike() respectively)
    """
    lp = lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(parameters, rossby_no, log_LxLbol, err_ll)


def run_rossby_fit(start_p, data_rossby, data_ll, data_ull,
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
        args=[data_rossby,data_ll,data_ull])
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

def plot_rossby_log(samples,data_rossby,data_ll,data_ull,
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
    #ax.set_yscale('log')
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
    ax.plot(xl,rossby_model_log([sl_mcmc[1][1],to_mcmc[1][1],be_mcmc[1][1]],xl),
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
