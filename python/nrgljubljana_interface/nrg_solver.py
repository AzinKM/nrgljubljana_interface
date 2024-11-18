from triqs.gf import *
from triqs.operators import *
from h5 import *
from triqs.utility import mpi
from nrgljubljana_interface import Solver, MeshReFreqPts, hilbert_transform_refreq
import math, os, warnings
import numpy as np
from scipy import interpolate, integrate, special, optimize
from collections import OrderedDict


class nrg_solver:
    def __init__(self, mu = None, parameters = None, set_up_parameters= None, solve_parameters = None, model_parameters = None, solution_filename="solution.h5", checkpoint_filename="checkpoint.h5"):
        '''
        Parameters
        ----------
        parameters : dictionary
            General parameters
            
        set_up_parameters : dictionary
            Parameters for setting up the Solver
            
        solve_parameters : dictionary
            Solver parameters
            
        model_parameters : dictionary
            Model parameters which is a dic inside solve_parameters
        '''
        self._default_dics()
        if not (parameters is None):
            self.par_dic.update(parameters)
            
        if not (set_up_parameters is None):
            self.set_up_dic.update(set_up_parameters)
            
        if not (solve_parameters is None):
            self.sp_dic.update(solve_parameters)
        
        if not (model_parameters is None):
            self.mp_dic.update(model_parameters)
            
            
        self.S = Solver(**set_up_parameters)
        self.S.set_verbosity(self.par_dic['verbose_nrg'])
        self.mp_dic = {k: v for k, v in self.mp_dic.items() if v is not None}
        self.sp_dic["model_parameters"] = self.mp_dic
        
        self.mu = mu
        self.newG = lambda : self.S.G_w.copy()
        self.Gloc = self.newG()
        self.Gself = self.newG()
        self.Gloc_prev = self.newG()
        
        self.nr_blocks = lambda bgf : len([bl for bl in bgf.indices]) # Returns the number of blocks in a BlockGf object
        self.block_size = lambda bl : len(self.S.G_w[bl].indices[0])       # Matrix size of Green's functions in block 'bl'
        self.identity = lambda bl : np.identity(self.block_size(bl))       # Returns the identity matrix in block 'bl'
        self.index_range = lambda G : range(len(G.indices[0]))             # Index range of a GF
        self.solution_filename = solution_filename
        self.checkpoint_filename = checkpoint_filename
        
        # Additional quantities of interest
        self.observables = ["n_d", "n_d^2"]
        
        self.par_dic['verbose'] = self.par_dic.get('verbose', True) and mpi.is_master_node()         # output is only produced by the master node
        self.par_dic['store_steps'] = self.par_dic.get('store_steps', True) and mpi.is_master_node()
    
    
    def _default_dics(self):
        self.par_dic = {
            'n_loop' : 9,
            'occupancy_goal' : 0.8,
            'max_mu_adjust' : 1,
            'occup_method' : "adjust",
            'alpha' : 0.5,
            'Delta_min' : 1e-5,
            'dos' : "Bethe",
            'Bethe_unit' : 1.0,
            'eps_prev' : 1e-3,
            'eps_loc_imp' : 1e-3,
            'eps_occupancy' : 1e-2,
            'min_iter' : 5,
            'max_iter' : 50,
            'verbose' : True,            # show info messages during the iteration
            'verbose_nrg' : False,       # show detailed output from the NRG solver
            'store_steps' : True,        # store intermediate results to files (one subdirectory per iteration)
            'normalize_to_one' : True
        }
        
        self.set_up_dic = {
            'model': "SIAM",
            'symtype': "QS",
            'mesh_max': 10.0,
            'mesh_min': 1e-5,
            'mesh_ratio': 1.01
        }
        
        self.sp_dic = {
            "T": 1e-4, 
            "Lambda": 2.0, 
            "Nz": 4, 
            "Tmin": 1e-5,
            "keep": 10000,
            "keepenergy": 10.0, 
            "bandrescale": 10.0,   
        }
        
        self.mp_dic = {
            "U1": 2.0, 
            "B1": None, 
            "omega1": None, 
            "g1": None, 
            "n1": None 
        }
        
        
   # Adjust Im(Delta) so that the hybridisation strength is not too small for the NRG discretization 
    def _fix_hyb_function(self, Delta):
        Delta_fixed = Delta.copy()
        for bl in Delta.indices:
            for w in Delta.mesh:
                for n in range(self.block_size(bl)): # only diagonal parts
                    r = Delta[bl][w][n,n].real
                    i = Delta[bl][w][n,n].imag
                    Delta_fixed[bl][w][n,n] = r + 1j*(i if i<-self.par_dic['Delta_min'] else -self.par_dic['Delta_min'])
        # Possible improvement: re-adjust the real part so that the Kramers-Kronig relation is maintained
        return Delta_fixed
    
    def _ht(self, z):      # Initialize a function ht0 for calculating the Hilbert transform of the DOS
        if (self.par_dic['dos'] == "Bethe"):
            ht1 = lambda z: 2*(z-1j*np.sign(z.imag)*np.sqrt(1-z**2)) # Analytical expression for Hilbert transform of Bethe lattice DOS
            self.ht0 = lambda z: ht1(z/self.par_dic['Bethe_unit'])
        else:
            table = np.loadtxt(dos)
            global dosA
            dosA = Gf(mesh=MeshReFreqPts(table[:,0]), target_shape=[])
            for i, w in enumerate(dosA.mesh):
                dosA[w] = np.array([[ table[i,1] ]])
            self.ht0 = lambda z: hilbert_transform_refreq(dosA, z)
            
        ht = lambda z: self.ht0(z.real+1j*(z.imag if z.imag>0.0 else 1e-20)) # Fix problems due to causality violation
        
        return ht(z)
        
    # Calculate the local lattice GF and the hybridisation function for the effective impurity model
    # for a given self-energy function
    def _self_consistency(self, Sigma):
        for bl in self.Gloc.indices:
            for w in self.Gloc.mesh:
                for i in range(self.block_size(bl)):
                    for j in range(self.block_size(bl)): # assuming square matrix
                        if i == j:
                            self.Gloc[bl][w][i,i] = self._ht(w + self.mu - Sigma[bl][w][i,i]) # Hilbert-transform
                        else:
                            assert abs(Sigma[bl][w][i,j])<1e-10, "This implementation only supports diagonal self-energy"
                            self.Gloc[bl][w][i,j] = 0.0
                            
        Glocinv = self.Gloc.inverse()
        Delta = self.newG()
        for bl in Delta.indices:
            for w in Delta.mesh:
                Delta[bl][w] = (w+self.mu)*self.identity(bl) - Sigma[bl][w] - Glocinv[bl][w] # !!!
                
        return Delta       
    
    # Calculate a GF from hybridisation and self-energy
    def _calc_G(self, Delta, Sigma):
        G = self.newG()
        for bl in G.indices:
            for w in G.mesh:
                G[bl][w] = np.linalg.inv( (w + self.mu)*self.identity(bl) - Delta[bl][w] - Sigma[bl][w] ) # !!!
        return G
    
    # Return an interpolation-object representation of a spectral function for GF G
    def _interp_A(self, G):
        lx = np.array(list(G.mesh.values()))
        ly = sum( sum( -(1.0/math.pi)*np.array(G[bl].data[:,i,i].imag) for i in self.index_range(G[bl]) ) for bl in G.indices )                                                                      # sum over blocks
        if self.par_dic['normalize_to_one']:
            nr = sum( sum( 1 for i in self.index_range(G[bl]) ) for bl in G.indices )                       # number of contributions
            ly = ly/nr                                                                                 # rescale
        return interpolate.interp1d(lx, ly, kind='cubic', bounds_error=False, fill_value=0)
    
    # Calculate occupancy for given hybridisation, self-energy and chemical potential
    def _calc_occupancy(self, Delta, Sigma, mu):
        self.mu = mu
        Gtrial = self._calc_G(Delta, Sigma)
        f = self._interp_A(Gtrial)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n = integrate.quad(lambda x : 2*f(x)*special.expit(-x/self.sp_dic['T']), -self.set_up_dic['mesh_max'], self.set_up_dic['mesh_max'])
        return n[0]

   # Difference between two Green's functions evaluated as the integrated squared difference between the
   # corresponding spectral functions. 
    def _gf_diff(self, a, b):
        f_a = self._interp_A(a)
        f_b = self._interp_A(b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diff = integrate.quad(lambda x : (f_a(x)-f_b(x))**2, -self.set_up_dic['mesh_max'], self.set_up_dic['mesh_max'])
        return diff[0]

    # Update mu towards reaching the occupancy goal
    def _update_mu(self, Delta, Sigma):
        mu = self.mu
        def F(x):
            density = self._calc_occupancy(Delta, Sigma, x)
            #print("update mu= ", x," n= ", density)
            return density - self.par_dic['occupancy_goal']
        sol = optimize.root_scalar(lambda x : F(x), x0=mu, x1=mu-0.1)
        self.mu = sol.root
        self.sp_dic["model_parameters"]["eps1"] = -self.mu
        
    # Iteratively adjust mu, taking into account the self-consistency equation.
    # Returns an improved estimate for the hybridisation function.
    def _adjust_mu(self, Delta_in, Sigma):
        old_mu = self.mu
        Delta = Delta_in.copy()
        for _ in range(self.par_dic['max_mu_adjust']):
            self._update_mu(Delta, Sigma)

            Delta = self._self_consistency(Sigma)
            
        new_mu = self.mu
        if self.par_dic['verbose']: print("Adjusted mu from %f to %f\n" % (old_mu,new_mu))
        return  Delta
      
    
    # Exception to raise when convergence is reached
    class Converged(Exception):
        def __init__(self, message):
            self.message = message

    # Exception to raise when convergence is not reached (e.g. maximum nr of iterations exceeded)
    class FailedToConverge(Exception):
        def __init__(self, message):
            self.message = message
            
    # Save a Green's function to a tabulated ASCII file
    def _save_Gf(self, fn, gf):
        f = open(fn, "w")
        for w in gf.mesh:
            z = gf[w]
            f.write("%f %f %f\n" % (w, z.real, z.imag))
    
    # Save all blocks for a block GF to tabulated ASCII files
    def _save_BlockGf(self, fn, bgf):
        for bl in bgf.indices:
            self._save_Gf(fn + "_" + bl + ".dat", bgf[bl])
    
    # Save a spectral function (-1/Pi Im GF) to a tabulated ASCII file
    def _save_A(self, fn, gf):
        f = open(fn, "w")
        for w in gf.mesh:
            z = gf[w]
            f.write("%f %f\n" % (w, -1.0/math.pi*z.imag))
    
    # Save spectral functions for all blocks of the block GF
    def _save_BlockA(self, fn, bgf):
        for bl in bgf.indices:
            self._save_A(fn + "_" + bl + ".dat", bgf[bl])
    
    # Store the complete set of results
    def _store_result(self, fn):
        with HDFArchive(fn, 'w') as arch:
            arch["S"] = self.S
            # Global variables
            arch["Gself"] = self.Gself
            arch["Gloc"] = self.Gloc
            arch["mu"] = self.mu
            
    # Formatting of the header in stats.dat
    def _fmt_str_header(self, nr_val):
        str = "{:>5}" # itern
        for _ in range(nr_val-1): str += " {:>15}"
        return str + "\n"

    # Formatting of the results in stats.dat
    def _fmt_str(self, nr_val):
        str = "{:>5}" # itern
        for _ in range(nr_val-1): str += " {:>15.8g}"
        return str + "\n"
    
    # Perform a DMFT step. Input is the hybridization function for solving the effective impurity model,
    # output is a new hybridization function resulting from the application of the DMFT self-consistency equation.
    def _dmft_step(self, Delta_in):
        Delta_in_fixed = self._fix_hyb_function(Delta_in)
        self.S.Delta_w << Delta_in_fixed
        self.S.solve(**self.sp_dic)
        self.Gself = self._calc_G(Delta_in_fixed, self.S.Sigma_w) # impurity GF ("self-energy-trick" improved)
        Delta = self._self_consistency(self.S.Sigma_w)     # apply the DMFT self-consistency equation
      
        diff_loc_imp = self._gf_diff(self.Gself, self.Gloc)            # difference between impurity and local lattice GF
        diff_prev = self._gf_diff(self.Gloc, self.Gloc_prev)           # difference between two consecutively computed local latice GFs
        self.Gloc_prev = self.Gloc.copy()
        occupancy = self._calc_occupancy(Delta, self.S.Sigma_w, self.mu)
        diff_occupancy = abs(occupancy-self.par_dic['occupancy_goal']) # this difference is used as the measure of deviation
        
        if self.par_dic['occup_method'] == "adjust":
            Delta = self._adjust_mu(Delta, self.S.Sigma_w) # here we update mu to get closer to target occupancy
            
        return Delta, diff_loc_imp, diff_prev, occupancy, diff_occupancy
    

    def solve_from_scratch(self):
        ###initialization
        Sigma0 = self.newG()
        for bl in Sigma0.indices:
            for w in Sigma0.mesh:
                Sigma0[bl][w] = self.mp_dic['U1']*self.par_dic['occupancy_goal']/2.0  # Initialize self-energy with the Hartree shift
        self.mu = self.mp_dic['U1']/2.0 # initial approximaiton for the chemical potential
        self.sp_dic["model_parameters"]["eps1"] = -self.mu
        Delta = self._self_consistency(Sigma0)
        Delta = self._adjust_mu(Delta, Sigma0)
        self.Gloc_prev = self.Gloc.copy()
        
        Delta_in = Delta.copy()
        for it in range(1, self.par_dic['n_loop']):
            Delta_out, diff_loc_imp, diff_prev, occupancy, diff_occupancy = self._dmft_step(Delta_in)
            newDelta = self.par_dic['alpha']*Delta_out + (1-self.par_dic['alpha'])*Delta_in
            Delta_in << newDelta
            
            
            stats = OrderedDict([("itern", it), ("mu", self.mu), ("diff_loc_imp", diff_loc_imp), ("diff_prev", diff_prev),
                       ("diff_occupancy", diff_occupancy), ("occupancy", occupancy)])
            for i in self.observables:
                stats[i] = self.S.expv[i]
            header_string = self._fmt_str_header(len(stats)).format(*[i for i in stats.keys()])
            stats_string  = self._fmt_str(len(stats)).format(*[i for i in stats.values()])
            if mpi.is_master_node():
                stats_file = open("stats.dat", "w", buffering=1) # line buffered
                if it == 1: stats_file.write(header_string)
                stats_file.write(stats_string)
            if self.par_dic['verbose']: print("stats: %sstats: %s" % (header_string, stats_string))
        
            if self.par_dic['store_steps']:
                os.mkdir(str(it)) # one subdirectory per iteration
                self._save_BlockGf(str(it)+"/Delta", Delta_in)
                self._save_BlockGf(str(it)+"/Sigma", self.S.Sigma_w) # self-energy
                self._save_BlockGf(str(it)+"/G", self.Gloc)          # local lattice Green's function
                self._save_BlockA(str(it)+"/A", self.Gloc)           # spectral function of local lattice GF
                self._store_result(str(it)+"/"+self.solution_filename)
        
            if mpi.is_master_node():
                self._store_result(self.checkpoint_filename) # for checkpoint/restart functionality
            

            # Check for convergence. The only way to exit the DMFT loop is by generating exceptions.
            if (diff_loc_imp   < self.par_dic['eps_loc_imp']   and
                diff_prev      < self.par_dic['eps_prev']      and
                diff_occupancy < self.par_dic['eps_occupancy'] and
                it >= self.par_dic['min_iter']):
                raise Converged(stats_string)
            if (it == self.par_dic['max_iter']):
                raise FailedToConverge(stats_string)
                
        mpi.barrier()
