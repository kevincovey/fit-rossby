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

def dual_power_law(parameters,x):
    """
    computes a dual-power law model

    For x >= turnover, the model values follow a power-law with slope beta_2:

            y = C + beta_2 log10(x)

    For x < turnover, the model values are a second power-law with slope beta_1:

            y = C + (beta_2 - beta_1)*log10(turnover) + beta_1 * log10(x)

    Inputs and outputs are in log space (ie, saturation level is -3., rather than 10.**(-3.); similar for loglxlbol values)

    Input
    -----
    parameters : array-like (4)
        parameters for the model: C (intercept constant), turnover, beta_1, beta_2

    Ro : array-like 
        Rossby number values. The model Log L_{whatever}/L_{bol} values will 
        be computed for these Rossby numbers

    Output
    ------
     : numpy.ndarray (same size as Ro)
        Model Log L_{whatever}/L_{bol} values corresponding to input Ro

    """

    #save the parameters with intuitive names
    intercept_constant, turnover, beta_1, beta_2 = parameters[0], parameters[1], parameters[2], parameters[3]

    #calculate the pivot constant that ensures the two laws meet at the same point
    pivot_constant = intercept_constant + (beta_2 - beta_1) * np.log10(turnover)

    #define the Log_LxLbol array and fill with saturated level datapoints
    Log_LxLbol = np.ones(len(x))

    #find unsaturated objects and calculate their Log_LxLbols based on the assumed power law behavior
    un_sat = np.where(x>=turnover)[0]
    Log_LxLbol[un_sat] = intercept_constant + beta_2 * np.log10(x[un_sat])

    #find saturated points and calculate their Log_LxLbols
    sat = np.where(x<turnover)[0]
    Log_LxLbol[sat] = pivot_constant + beta_1 * np.log10(x[sat])

    return Log_LxLbol

def dual_lnprior_periods_fixSlope(parameters, low_slope, high_slope):
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
    #print('slope bounds are: ', low_slope, high_slope)
    intercept_constant, turnover, beta_1, beta_2, lnf = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
    if 20 < intercept_constant < 40 and 2 < turnover < 50 and -4 < beta_1 < 2 and  low_slope < beta_2 < high_slope and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

def dual_lnprior_periods(parameters):
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
    
    intercept_constant, turnover, beta_1, beta_2, lnf = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
    if 20 < intercept_constant < 40 and 2 < turnover < 50 and -4 < beta_1 < 2 and  -5 < beta_2 < 1 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

def dual_lnprior(parameters):
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
    
    intercept_constant, turnover, beta_1, beta_2, lnf = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
    if -99 < intercept_constant < 100 and 0.05 < turnover < 0.5 and -3 < beta_1 < 2 and  -3 < beta_2 < 1 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf


def dual_lnlike(parameters, rossby_no, log_LxLbol ,err_ll): 
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
    
    intercept_constant, turnover, beta_1, beta_2, lnf = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
    #if ((sat_level>1e-1) or (sat_level<1e-8) or (turnover<0.001)    ## stephanie's original method of setting priors;
    #    or (turnover>2) or (beta>2) or (beta<-6)):                  ## now offloaded to lnprior
    #    return -np.inf

    model_ll = dual_power_law(parameters, rossby_no)

    #inv_sigma2 = 1.0/(err_ll**2)  ## inverse sigma assuming only quoted errors
    inv_sigma2 = 1.0/(err_ll**2 + model_ll**2*np.exp(2*lnf))  ## inverse sigma assuming errors are underestimated by some multiplicative factor
    ln_like = -0.5*(np.sum((log_LxLbol-model_ll)**2*inv_sigma2 - np.log(inv_sigma2)))

    return ln_like

def dual_lnprob_periods_fixed(parameters, rossby_no, log_LxLbol, err_ll, lowSlope, highSlope):
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

    lp = dual_lnprior_periods_fixSlope(parameters, lowSlope, highSlope)
    if not np.isfinite(lp):
        return -np.inf
    return lp + dual_lnlike(parameters, rossby_no, log_LxLbol, err_ll)


def dual_lnprob_periods(parameters, rossby_no, log_LxLbol, err_ll):
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

    lp = dual_lnprior_periods(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + dual_lnlike(parameters, rossby_no, log_LxLbol, err_ll)


def dual_lnprob(parameters, rossby_no, log_LxLbol, err_ll):
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
    lp = dual_lnprior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp + dual_lnlike(parameters, rossby_no, log_LxLbol, err_ll)


def run_dual_fit(start_p, data_rossby, data_ll, data_ull,
    nwalkers=256,nsteps=40000):
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

    ndim = 5
    p0 = np.zeros((nwalkers,ndim))

    # initialize the walkers in a tiny gaussian ball around the starting point
    for i in range(nwalkers):
        p0[i] = start_p + (1e-1*np.random.randn(ndim)*start_p)

    sampler = emcee.EnsembleSampler(nwalkers,ndim,dual_lnprob,
        args=[data_rossby,data_ll,data_ull])
    pos,prob,state=sampler.run_mcmc(p0,nsteps/2)
    sampler.reset()
    pos,prob,state=sampler.run_mcmc(pos,nsteps)

    ic_mcmc = quantile(sampler.flatchain[:,0],[.16,.5,.84])

    #sl_mcmc.info()
    #print(sl_mcmc)
    to_mcmc = quantile(sampler.flatchain[:,1],[.16,.5,.84])
    #print(to_mcmc)
    beta1_mcmc = quantile(sampler.flatchain[:,2],[.16,.5,.84])
    beta2_mcmc = quantile(sampler.flatchain[:,3],[.16,.5,.84])
    #print(be_mcmc)
    var_mcmc = quantile(sampler.flatchain[:,4],[.16,.5,.84])

    
    print('intercept constant={0:.7f} +{1:.7f}/-{2:.7f}'.format(
        ic_mcmc[1][1],ic_mcmc[1][1]-ic_mcmc[0][1],ic_mcmc[2][1]-ic_mcmc[1][1]))
    print('turnover={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        to_mcmc[1][1],to_mcmc[1][1]-to_mcmc[0][1],to_mcmc[2][1]-to_mcmc[1][1]))
    print('beta1={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        beta1_mcmc[1][1],beta1_mcmc[1][1]-beta1_mcmc[0][1],beta1_mcmc[2][1]-beta1_mcmc[1][1]))
    print('beta2={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        beta2_mcmc[1][1],beta2_mcmc[1][1]-beta2_mcmc[0][1],beta2_mcmc[2][1]-beta2_mcmc[1][1]))
    print('var={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        var_mcmc[1][1],var_mcmc[1][1]-var_mcmc[0][1],var_mcmc[2][1]-var_mcmc[1][1]))

    samples = sampler.flatchain

    return samples

def run_dual_fit_periods_constrained(start_p, data_rossby, data_ll, data_ull, lowSlope, highSlope,
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
    ndim = 5
    p0 = np.zeros((nwalkers,ndim))

    # initialize the walkers in a tiny gaussian ball around the starting point
    for i in range(nwalkers):
        p0[i] = start_p + (1e-1*np.random.randn(ndim)*start_p)

    sampler = emcee.EnsembleSampler(nwalkers,ndim,dual_lnprob_periods_fixed,
        args=[data_rossby,data_ll,data_ull,lowSlope,highSlope])
    pos,prob,state=sampler.run_mcmc(p0,nsteps/2)
    sampler.reset()
    pos,prob,state=sampler.run_mcmc(pos,nsteps)

    ic_mcmc = quantile(sampler.flatchain[:,0],[.16,.5,.84])

    #sl_mcmc.info()
    #print(sl_mcmc)
    to_mcmc = quantile(sampler.flatchain[:,1],[.16,.5,.84])
    #print(to_mcmc)
    beta1_mcmc = quantile(sampler.flatchain[:,2],[.16,.5,.84])
    beta2_mcmc = quantile(sampler.flatchain[:,3],[.16,.5,.84])
    #print(be_mcmc)
    var_mcmc = quantile(sampler.flatchain[:,4],[.16,.5,.84])

    
    print('intercept constant={0:.7f} +{1:.7f}/-{2:.7f}'.format(
        ic_mcmc[1][1],ic_mcmc[1][1]-ic_mcmc[0][1],ic_mcmc[2][1]-ic_mcmc[1][1]))
    print('turnover={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        to_mcmc[1][1],to_mcmc[1][1]-to_mcmc[0][1],to_mcmc[2][1]-to_mcmc[1][1]))
    print('beta1={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        beta1_mcmc[1][1],beta1_mcmc[1][1]-beta1_mcmc[0][1],beta1_mcmc[2][1]-beta1_mcmc[1][1]))
    print('beta2={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        beta2_mcmc[1][1],beta2_mcmc[1][1]-beta2_mcmc[0][1],beta2_mcmc[2][1]-beta2_mcmc[1][1]))
    print('var={0:.3f} +{1:.3f}/-{2:.3f}'.format(
        var_mcmc[1][1],var_mcmc[1][1]-var_mcmc[0][1],var_mcmc[2][1]-var_mcmc[1][1]))

    samples = sampler.flatchain

    return samples

def plot_dual_fit(samples,data_rossby,data_ll,data_ull,plotfilename=None,ylabel=r'$L_{X}/L_{bol}$', sampleName=None):
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
    
    ic_mcmc = quantile(samples[:,0],[.16,.5,.84])
    to_mcmc = quantile(samples[:,1],[.16,.5,.84])
    beta1_mcmc = quantile(samples[:,2],[.16,.5,.84])
    beta2_mcmc = quantile(samples[:,3],[.16,.5,.84])
    var_mcmc = quantile(samples[:,4],[.16,.5,.84])

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    # Just trying to reduce the number of plotted points...    
    xl = np.append(np.arange(0.001,0.2,0.001),np.arange(0.2,2.5,0.02))
#    xl = np.arange(0.001,2.0,0.005)
    #for p in list(samples[np.random.randint(len(samples), size=100)]):
    #    ax.plot(xl,rossby_model(p,xl),color='LightGrey')

    intercept_constant = ic_mcmc[1][1]
    turnover = to_mcmc[1][1]
    x = np.asarray([turnover,2.0])
#    x = np.arange(turnover,2.0,0.001)
    #constant = sat_level/(turnover**-1.)
    #ax.plot(x,constant*(x**-1.),'k--',lw=1.5,label=r'$\beta=\ -1$')
    #constant = sat_level/(turnover**-2.1)
    #ax.plot(x,constant*(x**-2.1),'k-.',lw=1.5,label=r'$\beta=\ -2.1$')
    #constant = sat_level/(turnover**-2.7)
    #ax.plot(x,constant*(x**-2.7),'k:',lw=2,label=r'$\beta=\ -2.7$')

    star_color = 'steelblue'
    ax.errorbar(data_rossby,data_ll,data_ull,color=star_color,fmt='.',capsize=1,
        ms=2,mec=star_color)
    #print('parameters for model plot:')
    #print('xl: ')
    #print(xl)
    #print('model inputs: ')
    #print([sl_mcmc[1][1],to_mcmc[1][1],be_mcmc[1][1]])
    #print('model: ')
    #print(
    ax.plot(xl,dual_power_law([ic_mcmc[1][1],to_mcmc[1][1],beta1_mcmc[1][1],beta2_mcmc[1][1]],xl),
        'k-',lw=2,label=r'$\beta1=\ {0:.1f}$'.format(beta1_mcmc[1][1]))
    ax.set_ylabel(ylabel,fontsize='xx-large')
    ax.set_xlabel('R$_o$',fontsize='x-large')
    ax.set_xlim(1e-3,2)
    ax.tick_params(labelsize='x-large')
    #ax.set_xticklabels((0.001,0.01,0.1,1))

    handles, labels = ax.get_legend_handles_labels()
    new_handles = np.append(handles[-1],handles[0:-1])
    new_labels = np.append(labels[-1],labels[0:-1])
    if sampleName!=None:
        ax.legend(new_handles,new_labels,loc=3, title=sampleName)
    else:
        ax.legend(new_handles,new_labels,loc=3)

    if plotfilename!=None:
        plt.savefig(plotfilename)

def plot_dual_fit_periods(samples,data_rossby,data_ll,data_ull,plotfilename=None,ylabel=r'$Log L_{X}$', sampleName=None):
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

    #print(len(data_rossby),len(data_ll), len(data_ull))
    
    ic_mcmc = quantile(samples[:,0],[.16,.5,.84])
    to_mcmc = quantile(samples[:,1],[.16,.5,.84])
    beta1_mcmc = quantile(samples[:,2],[.16,.5,.84])
    beta2_mcmc = quantile(samples[:,3],[.16,.5,.84])
    var_mcmc = quantile(samples[:,4],[.16,.5,.84])

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    # Just trying to reduce the number of plotted points...    
    xl = np.append(np.arange(0.05,7,0.01),np.arange(7,160,0.5))
#    xl = np.arange(0.001,2.0,0.005)
    #for p in list(samples[np.random.randint(len(samples), size=100)]):
    #    ax.plot(xl,rossby_model(p,xl),color='LightGrey')

    intercept_constant = ic_mcmc[1][1]
    turnover = to_mcmc[1][1]
    x = np.asarray([turnover,2.0])
#    x = np.arange(turnover,2.0,0.001)
    #constant = sat_level/(turnover**-1.)
    #ax.plot(x,constant*(x**-1.),'k--',lw=1.5,label=r'$\beta=\ -1$')
    #constant = sat_level/(turnover**-2.1)
    #ax.plot(x,constant*(x**-2.1),'k-.',lw=1.5,label=r'$\beta=\ -2.1$')
    #constant = sat_level/(turnover**-2.7)
    #ax.plot(x,constant*(x**-2.7),'k:',lw=2,label=r'$\beta=\ -2.7$')

    star_color = 'steelblue'
#    ax.errorbar(data_rossby,data_ll,data_ull,color=star_color,fmt='.',capsize=0,
#        ms=4,mec=star_color)
    ax.scatter(data_rossby,data_ll,color=star_color) #,fmt='.',capsize=0,
 #       ms=4,mec=star_color)
    #print('parameters for model plot:')
    #print('xl: ')
    #print(xl)
    #print('model inputs: ')
    #print([sl_mcmc[1][1],to_mcmc[1][1],be_mcmc[1][1]])
    #print('model: ')
    #print(
    ax.plot(xl,dual_power_law([ic_mcmc[1][1],to_mcmc[1][1],beta1_mcmc[1][1],beta2_mcmc[1][1]],xl),
        'k-',lw=2,label=r'$\beta1=\ {0:.2f}$'.format(beta1_mcmc[1][1])+"\n"+r'$\beta2=\ {0:.2f}$'.format(beta2_mcmc[1][1]) )
    ax.set_ylabel(ylabel,fontsize='xx-large')
    ax.set_xlabel(r'P$_{rot}$',fontsize='x-large')
    ax.set_xlim(0.05,200)
    ax.tick_params(labelsize='x-large')
    #ax.set_xticklabels((0.001,0.01,0.1,1))

    handles, labels = ax.get_legend_handles_labels()
    new_handles = np.append(handles[-1],handles[0:-1])
    new_labels = np.append(labels[-1],labels[0:-1])
    if sampleName!=None:
        ax.legend(new_handles,new_labels,loc=3, title=sampleName)
    else:
        ax.legend(new_handles,new_labels,loc=3)

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
