#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:15:17 2022

@author: J Andrés Christen, jac at cimat.mx

Posterior sampling using the twalk:

Templete class, pyPstwalk, to perform Byesian inference using the t-walk.

See below.

"""

from time import localtime, strftime

from numpy import array, ceil, zeros, linspace, loadtxt, savetxt, arange, exp, where
from numpy import sum as np_sum
from scipy.stats import bernoulli, weibull_min, norm, gamma, expon
from matplotlib.pylab import subplots

from pytwalk import pytwalk

class pyPstwalk:
    """
        pyPstwalk: pyThon class to perform Psterior sampling (for Bayesian
        inference) using the t-walk:
        
        This is a template class, a derived class needs to be coded to define
        the loglikehood.  See below the examples for independent 1D sampling,
        ind1Dsampl, and for Bernoulli-Weibull regresion .
        
        par_names: a list with the parameter names, in the correct order.
           Internally, an array will hold the values for this parameters and
           this defines in which order; the same for par_prior and par_supp.
        par_prior: a list with objects definig the marginal prior of each
           parameter.  logprior assumes independence, and this may be changed
           by overloading this method.  The objects defining the priors most
           have the method logpdf. Typically a 'frozen' distribution from
           scipy.stats (see examples below).  Also, it should have the method
           rvs to simulate initial values, from the prior, for the t-walk.
        par_supp: a list of callables, taking a float and returning True or False
           to define if the corresponding parameters is its support.
        default_burn_in: a default vlaue for the burn_in.
        default_op_fnam: a default name for a csv file to save the twalk.Output
           array to.  This is automatically done after the t-walk finishes and
           may be retrived with LoadtwalkOuput.
    """
    
    def __init__(self, par_names, par_prior, par_supp, default_burn_in=0, default_op_fnam = "pyPstwalk.csv"):
        
        self.par_names = par_names
        ### Create a string with the par names separated by commas, for the csv file
        ### to save twalk.Output
        self.col_names = ""
        for pn in self.par_names:
            self.col_names += '"' + pn + '", '
        self.col_names += '"Energy"' #last colomun, -logpost
        self.q = len(self.par_names) #Number of parameters
        ###                 a                      b                 al              beta
        self.par_prior = par_prior        
        self.par_supp  = par_supp
        
        self.twalk = pytwalk(  n=self.q, U=self.Energy, Supp=self.Supp)
        self.default_burn_in = default_burn_in
        
    def fpi( self, pname):
        """Find the index of parameter with (possible partial) name pname.
           returns -1 no matches, < .1 several matches or the index.
        """
        tmp = where([pn.count(pname)>0 for pn in self.par_names])[0]
        if tmp.size != 1:
            ### pnam not found or several matches found
            return -tmp.size -1 #-1 no matches, < .1 several matches
        else:
            return tmp[0]
    
    def loglikelihood(self, x):
        """Overload this method to define the likelihood.
             x is an array containing the parameters in the order spacified
             by self.par_names.  One may translate this first to define the
             loglikelihood, e.g:
            a, b, c = x
        """
        pass
    
    def logprior(self, x):
        """The log prior assumes independece and multiplies the marginal prior
            for each parameter.
        """
        return sum([prior.logpdf(x[i]) for i,prior in enumerate(self.par_prior)])
    
    def Supp(self, x):
        """Checks if all parameters are in the support, evaluating supp."""
        return all([supp(x[i]) for i,supp in enumerate(self.par_supp)])
    
    def SimInit(self):
        """Simulates a initial value for the t-walk from the prior."""
        return array([prior.rvs() for prior in self.par_prior])
    
    def Energy( self, x):
        """-lop of the posterior."""
        return -1*(self.loglikelihood(x) + self.logprior(x))
        
    def RunMCMC( self, T, burn_in=-1, op_fnam=None):
        """Runs the t-walk for T iterations.  AnaMCMC is called to calculate
             the IAT with a burn in of burn_in.  op_fnam is an optional
             file name to save twalk.Output to.
        """
        self.twalk.Run( T=T, x0=self.SimInit(), xp0=self.SimInit())
        self.T = self.twalk.Output.shape[0]
        if self.default_burn_in == -1:
            burn_in = self.default_burn_in
        else:
            self.default_burn_in = burn_in
        if op_fnam is not None:
            savetxt( op_fnam, self.twalk.Output, fmt='%.15g', comments='#',\
                header="PyPstwalk: twalk Output, finished: %s\n" % (strftime( "%Y.%m.%d:%H:%M:%S", localtime()))\
                         + self.col_names)
            print("PyPstwalk: twalk output saved in %s." % (op_fnam))
        self.AnaMCMC(burn_in)
    
    def LoadtwalkOuput(self, op_fnam):
        """Load the twak.Output."""
        self.twalk.Output = loadtxt(op_fnam)
        self.T = self.twalk.Output.shape[0]
        
    def AnaMCMC(self, burn_in):
        """Calculate the iat and some other analysis calling twalk.Ana."""
        self.iat = int(ceil(self.twalk.Ana(start=burn_in)))

    def PlotTs( self, par=-1, burn_in=-1, iat=1, ax=None, **kargs):
        """Plot a time series of parameter (number) par. par may also be the (partial) name
           of the parameter (string).
           par=-1 plots the logpost (not the Energy). 
           The plot is done in the axis ax, if None, one is created and returned.
           All other kargs are passed to ax.plot .
           If burn_in = -1, self.default_burn_in is used.
        """
        if isinstance(par,str):
            tmp = self.fpi(par)
            if tmp == -1:
                print("PyPstwalk:PlotPost: par. name %s not found." % (par))
                return None
            elif tmp < -1:
                print("PyPstwalk:PlotPost: %d matches found for par. name %s." % (-tmp-1,par))
                return None
            else:
                par = tmp # par index
        if burn_in == -1:
            burn_in = self.default_burn_in
        else:
            self.default_burn_in = burn_in
        if ax is None:
            fig, ax = subplots()
        t = arange( burn_in, self.T)
        if par == -1:
            label = "LogPost"
            mult = -1
        else:
            label = self.par_names[par]
            mult = 1
        ax.plot( t, mult*self.twalk.Output[burn_in:self.T:iat,par], '-', **kargs)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(label)
        return ax

    def PlotPost( self, par, burn_in=-1, ax=None,\
                 prior_color="green", density=True, **kargs):
        """Plot a histogram of parameter (number) par. par may also be the (partial) name
           of the parameter (string).
           The plot is done in the axis ax, if ax=None, one is created and returned.
           All other kargs are passed to ax.hist .
           If burn_in = -1, self.default_burn_in is used.
           
           The prior is also plotted within the bounds of the posterior (it may
            appear as a line only etc.), with prior_color.  If prior_color=None
           then the prior is not plotted.
        """
        if isinstance(par,str):
            tmp = self.fpi(par)
            if tmp == -1:
                print("PyPstwalk:PlotPost: par. name %s not found." % (par))
                return None
            elif tmp < -1:
                print("PyPstwalk:PlotPost: %d matches found for par. name %s." % (-tmp-1,par))
                return None
            else:
                par = tmp # par index
        if burn_in == -1:
            burn_in = self.default_burn_in
        else:
            self.default_burn_in = burn_in
        if ax is None:
            fig, ax = subplots()
        ax.hist( self.twalk.Output[burn_in:,par], density=density, **kargs)
        ax.set_xlabel(self.par_names[par])
        if prior_color is not None:
            xl = ax.get_xlim()
            x = linspace( xl[0], xl[1], num=200)
            ax.plot( x, exp(self.par_prior[par].logpdf(x)), '-', color=prior_color)
            ax.set_xlim(xl)
        return ax


class ind1Dsampl(pyPstwalk):
    """Dereived class from pyPstwalk for 1D independent sampling.
         q: numper of unknown parameters.
         data: array with data (=None run with no data).
         logdensity: log of the density, with signature logdensity( data, x),
          evaluates the log of the density at data (needs to be vectorized) for
          parameter x.  See Gamma Sampling example below.
         par_names, par_prior, par_supp, default_op_fnam: as in pyPstwalk
         simdata: optional, to simulate data with signature simdata( n, x), to
          simulate a sample of size n with 'true' parameters x.
    
    """
    
    def __init__( self, q, data, logdensity, par_names, par_prior, par_supp, simdata=None, default_op_fnam = "pyPstwalk_ind1Dsampl.csv"):
        
        self.logdensity = logdensity
        self.data = data
        if data is None:
            self.n = 0
        else:
            self.n = self.data.size
        self.simdata = simdata
        
        super().__init__(par_names=par_names,\
                         par_prior=par_prior,\
                         par_supp =par_supp, default_op_fnam = default_op_fnam)
            
    def loglikelihood(self, x):
        if self.n == 0:
            return 0.0 ## Run with no data, ie simulate from the prior
        else:
            return np_sum( self.logdensity( self.data, x))
    
    def SimData(self, n, x):
        self.data = self.simdata( n, x)
        self.n = self.data.size



class berreg(pyPstwalk):
    
    """Bernoulli regresion, dereived class from pyPstwalk for 1D independent sampling. 
        Bernoulli regression:
        $$
        X_{i,j} | \theta, a, b  \sim Ber( \Phi_{\theta} ( a + t_j b))
        $$
        where $\Phi$ is a pdf, default Weibull.
        
        t: array of lenght $m$ with times $t_j$
        data: nxm array with data.
    """
    
    def __init__( self, t, data, dist=weibull_min):
        
        self.t = t
        self.m = t.size
        self.data = data
        self.dist = dist
        self.n = self.data.shape[0]
        
        super().__init__(par_names=[              r"$a$",             r"$b$",        r"$\alpha$",       r"$\beta$"],\
                         par_prior=[norm(loc=0, scale=1), gamma( 3, scale=1), gamma( 3, scale=1), gamma( 3, scale=1)],\
                         par_supp =[      lambda x: True,    lambda x: x>0.0,    lambda x: x>0.0,   lambda x: x>0.0])
            
    def loglikelihood(self, x):
        a, b, al, beta = x
        ll = 0.0
        for i in range(self.n):
            ll += np_sum( bernoulli.logpmf( self.data[i,:],\
                                self.dist.cdf( a+b*self.t, al, scale=beta)))
        return ll


class BUQ(pyPstwalk):
    """Dereived class from pyPstwalk for simple Bayesian Inverse problems (Bayesian UQ).
       $$
        E(y_i | \theta) = F_\theta(t_j)
        y_i | \theta, \simga  \sim G(\sigma)
       $$
       independent.
         q: number of unknown parameters, card of theta + sigma,
            if sigma (dispersion parameters) also unknown.
         sigma: values for the disperssion parameter if known, if sigma=None,
          then it is assumed not known and is the last parameter in the parameter
          list, ie: "x = theta + sigma"
         data: array with data (=None run with no data).
         logdensity: log of the density of G, with signature
            logdensity( data, loc, scale),
          evaluates the log of the density at data (needs to be vectorized) for
          parameter x.
            simdata( n, loc, scale), to
          simulate a sample of size n with 'true' parameters loc=theta and scale=sigma.
          
        F: forward map, with signature F( theta, t), vectorized in t.
        t: localtions (eg. time) to evaluate the Forwad map.

        par_names, par_prior, par_supp, default_op_fnam: as in pyPstwalk
         simdata: optional, to simulate data from G with signature
    
    """
    
    def __init__( self, q, data, logdensity, sigma, F, t,\
                    par_names, par_prior, par_supp, simdata=None, default_op_fnam = "pyPstwalk_ind1Dsampl.csv"):
        
        self.logdensity = logdensity
        self.t = t
        self.data = data
        if data is None:
            self.n = 0
        else:
            self.n = self.data.size
        self.simdata = simdata
        self.sigma = sigma
        if sigma is None:
            self.sigma_known = False
        else:
            self.sigma_known = True
        self.F = F
        
        super().__init__(par_names=par_names,\
                         par_prior=par_prior,\
                         par_supp =par_supp, default_op_fnam = default_op_fnam)
            
    def loglikelihood(self, x):
        if self.n == 0:
            return 0.0 ## Run with no data, ie simulate from the prior
        else:
            if self.sigma_known:
                return np_sum( self.logdensity( self.data, loc=self.F( x, self.t), scale=self.sigma))
            else:
                return np_sum( self.logdensity( self.data, loc=self.F( x[:-1], self.t), scale=x[-1]))
    
    def SimData(self, x):
        """Simulate data at points self.t"""
        self.par_true = x
        if self.sigma_known:
            #theta = x
            self.data = self.simdata( self.t.shape, loc=self.F( x, self.t), scale=self.sigma)
        else:
            #theta = x[:-1], sigma = x[-1]
            self.data = self.simdata( self.t.shape, loc=self.F( x[:-1], self.t), scale=x[-1])
        self.n = self.data.shape



if __name__ == "__main__":
    
    ex = ["Gamma sampling", "Bernoulli Regresion", "BUQ"]
    example = "BUQ" #"Gamma sampling" #"Bernoulli Regresion"
    
    if example == "BUQ":
        ### Define the Forward map with signature F( theta, t)
        def F( theta, t):
            """Simple analytic FM, for this example !"""
            al, la = theta
            return al + exp(la*t)
        t = linspace(0, 1, num=30) #The sample size is 30, a grid of size 30
        sigma = 0.1
        ### logdensity: log of the density of G, with signature logdensity( data, loc, scale)
        ### see docstring of BUQ
        logdensity=norm.logpdf
        simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
        par_names=[r"$\alpha$", r"$\lambda$"]
        par_prior=[ gamma( 3, scale=1), gamma(1.1, scale=1)]
        par_supp  = [ lambda al: al>0.0, lambda la: la>0.0]
        buq = BUQ( q=2, data=None, logdensity=logdensity, simdata=simdata, sigma=sigma,\
                  F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
        buq.SimData(x=array([3,0.1])) #True parameters alpha=3 lambda=0.1
        buq.RunMCMC( T=100000, burn_in=1000, op_fnam="buq_output.csv")
        buq.PlotPost(par=0)
        buq.PlotPost(par="lam") #we may acces the parameters by name also
    
    if example == "Gamma sampling":
        ### Example with Gamma data with both shape and *rate* parameters unknown
        q = 2 # Number of unkown parameters, ie dimension of the posterior
        ###                       x[0],     x[1]
        par_names = [       r"$\alpha$", r"$\beta$"] # shape and rate
        ### The piros are assumed independent and exponential
        ### I use 'frozen' distributions from scipy.stats,
        ### although only the methods logpdf and rvs are required
        par_prior = [    expon(scale=1), expon(scale=1)]
        ### The support (both positive) is defined like this:
        par_supp  = [ lambda al: al>0.0, lambda beta: beta>0.0]
        ### This is log of the density, needs to be vectorized, and a function (optional) to simulate data
        logdensity = lambda data, x: gamma.logpdf( data, a=x[0], scale=1/x[1])
        simdata    = lambda n, x: gamma.rvs(a=x[0], scale=1/x[1], size=n)
        
        gammasmpl = ind1Dsampl( q=q, data=None, logdensity=logdensity,\
                par_names=par_names, par_prior=par_prior, par_supp=par_supp, simdata=simdata)
        par_true = array([ 3, 10])
        gammasmpl.SimData( n=30, x=par_true) # Simulate data
        gammasmpl.RunMCMC( T=50000, burn_in=5000) #op_name=None, no ouput save
        ax_al = gammasmpl.PlotPost(par=0, density=True)
        ax_al.axvline( par_true[0], ymax=0.1, color="black") #alpha true value
        ax_beta = gammasmpl.PlotPost(par=1, density=True)
        ax_beta.axvline( par_true[1], ymax=0.1, color="black") #alpha true value
    
    if example == "Bernoulli Regresion":
        t = array([0, 10, 20 , 30, 40])
        m = t.size
        n = 10
        data = zeros((n,m))
        a=0
        b=0.04
        al = 2
        beta = 1
        for i in range(n):
            data[i,:] = bernoulli.rvs( weibull_min.cdf( a+b*t, al, scale=beta))
        print(data)
    
        br = berreg( t, data)
        br.RunMCMC(T=50000, burn_in=1000)
        #br.LoadtwalkOuput()
        fig, ax = subplots(nrows=2,ncols=2)
        ax = ax.flatten()
        for i in range(4):
            br.PlotPost( par=i, ax=ax[i])
        """
        ##### Tostadas data:
        data_tostadas = loadtxt("tostadas.csv", skiprows=1)
        t_tostadas = array([0, 10, 20 , 30, 40])
        tostadas = berreg( t, data, defualt_burn_in=1000)
        """

