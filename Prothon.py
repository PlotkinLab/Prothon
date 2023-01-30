
### DESCRIPTION ###
# A Python package for efficient comparison of protein ensembles


import numpy as np
import matplotlib.pyplot as plt
from mdtraj import load, load_dcd, join 
from mdtraj import compute_angles, compute_dihedrals, compute_distances, shrake_rupley
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, mannwhitneyu
from glob import glob
from sys import exit
import warnings

# Silence scipy warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)


__author__ = "Aina Adekunle"
__copyright__ = "Copyright 2023, Aina Adekunle"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Aina Adekunle"
__email__ = "aina@phas.ubc.ca"
__status__ = "Production"



class Prothon():
    """Represent protein ensembles and calculate ensemble dissimilarity

    The Prothon class is initialized with a list of ensembles and the topology
    of the conformations contained in these ensembles. Method 
    ensemble_representation is used to represent ensembles using different 
    local structural properties and  method ensemble_dissimilarity is used to
    estimate the dissimilarity between ensembles.
    """
    
    def __init__(self, data, topology, verbose=False):
        """Initizlize the Prothon class 

        Parameters
        ----------
        data : List of strings
            List of filename(s) containing trajectory file(s) of a single format.
        topology: String
            Path to a PDB file of the topology of the conformations contained in data.
        verbose: bool, optional
            If it is set to be True, then the program will print out more 
            progress information as it runs.
        """

        # initialize class attributes
        self.data = data
        self.topology = topology
        #print(self.topology)
        self.verbose = verbose


    # C-beta contact number similarity measure
    def ensemble_cbcn(self, ensemble):
        """Ensemble representation in C-beta carbon contact numbers

        
        Parameters
        ----------
        ensemble : String
            Filename of an ensemble of protein conformations
        
            
        Returns
        -------
        CBCN : np.array, shape=(len(m), len(n))
            The C-beta carbon contact number for each amino acid residue n in each 
            conformation m contained in `ensemble`.
            
        """

        # Load ensemble
        if ensemble[-4:] == '.dcd':
            ensemble = load_dcd(ensemble, self.topology)
        else:
            ensemble = load(ensemble, self.topology) # try agnostic load 

        
        CONST = 50  # 1/nm
        r0 = 1      # nm ~= 3 CA-CA distance
        
        # Get the indices of all C-beta atoms
        cb_indices = ensemble.topology.select("name == 'CB'")
        #print(f'cb_indices::{len(cb_indices)}')
        
        # Get the pairs of C-beta atoms (i,j) in residues (a,b), where |a-b| > 2
        cb_pairs = np.array(
            [(i,j) for (i,j) in combinations(cb_indices, 2)
                if abs(ensemble.topology.atom(i).residue.index - \
                    ensemble.topology.atom(j).residue.index) > 2])
        
        # Calculate the CBCN for all residues for all conformations 
        cbcn = []

        for i, idx in enumerate(cb_indices):
            # Get C-beta pairs with index i present in pair   
            cbi_pairs = cb_pairs[[idx in pair for pair in cb_pairs]]

            # Compute pair distances involving C-beta with index i 
            r = compute_distances(ensemble, cbi_pairs)

            # Compute contact number for each C-beta atom
            n = np.sum(1.0 / (1 + np.exp(CONST * (r - r0))), axis=1)
            
            cbcn.append(n)

        return np.transpose(np.array(cbcn))



    # C-alpha contact number similarity measure
    def ensemble_cacn(self, ensemble):
        """Ensemble representation in C-alpha carbon contact numbers

        
        Parameters
        ----------
        ensemble : String
            Filename of an ensemble of protein conformations
        
            
        Returns
        -------
        CACN : np.array, shape=(len(m), len(n))
            The C-alpha carbon contact number for each amino acid residue n in each 
            conformation m contained in `ensemble`.
            
        """

        # Load ensemble
        if ensemble[-4:] == '.dcd':
            ensemble = load_dcd(ensemble, self.topology)
        else:
            ensemble = load(ensemble, self.topology) # try agnostic load 

        
        CONST = 50  # 1/nm
        r0 = 1      # nm ~= 3 CA-CA distance
        
        # Get the indices of all C-alpha atoms
        ca_indices = ensemble.topology.select("name == 'CA'")
        
        
        # Get the pairs of C-alpha atoms (i,j) in residues (a,b), where |a-b| > 2
        ca_pairs = np.array(
            [(i,j) for (i,j) in combinations(ca_indices, 2)
                if abs(ensemble.topology.atom(i).residue.index - \
                    ensemble.topology.atom(j).residue.index) > 2])
        
        # Calculate the CACN for all residues for all conformations 
        cacn = []

        for i, idx in enumerate(ca_indices):
            # Get C-beta pairs with index i present in pair   
            cai_pairs = ca_pairs[[idx in pair for pair in ca_pairs]]

            # Compute pair distances involving C-alpha with index i 
            r = compute_distances(ensemble, cai_pairs)

            # Compute contact number for each C-alpha atom
            n = np.sum(1.0 / (1 + np.exp(CONST * (r - r0))), axis=1)
            
            cacn.append(n)

        return np.transpose(np.array(cacn))





    # Virtual C-alpha - C-alpha bond angle similarity measure
    def ensemble_caba(self, ensemble):
        '''Ensemble representation in virtual CA-CA bond angle space
        
        Parameters
        ----------
        ensemble : String
            Filename of an ensemble of protein conformations
        
            
        Returns
        -------
        CABA : np.array, shape=(len(m), len(n-3))
            The C-alpha bond angle for each amino acid residue n in each 
            conformation m contained in `ensemble`.
            
        '''

        # Load ensembles
        if ensemble[-4:] == '.dcd':
            ensemble = load_dcd(ensemble, self.topology)
        else:
            ensemble = load(ensemble, self.topology) # agnostic load

        # Select C_alpha atoms and bond angle indices
        c_alphas = ensemble.topology.select("name == 'CA'")

        n = len(c_alphas)-3 # Number of bond angles

        bonds_indices = []
        
        for i in range(n): # change to list comprehension
            bonds_indices.append((c_alphas[i], c_alphas[i+1], c_alphas[i+2]))
           

        # Compute bond angles
        bonds = compute_angles(ensemble, bonds_indices)
    
        return bonds 


    # Virtual CA-CA torsion angle similarity measure
    def ensemble_cata(self, ensemble):
        '''Ensemble representation in virtual CA-CA dihedral angle space
        

        Parameters
        ----------
        ensemble : String
            Filename of an ensemble of protein conformations
        
            
        Returns
        -------
        CATA : np.array, shape=(len(m), len(n-4))
            The C-alpha torsion angle for each amino acid residue n in each 
            conformation m contained in `ensemble`.
            
        '''

        # Load ensembles
        if ensemble[-4:] == '.dcd':
            ensemble = load_dcd(ensemble, self.topology)
        else:
            ensemble = load(ensemble, self.topology) # agnostic load

        # Select C_alpha atoms and bond and dihedral angle indices
        c_alphas = ensemble.topology.select("name == 'CA'")

        n = len(c_alphas)-3 # Number bond angles and dihedral angles, respectively

        dihedral_indices = []
        for i in range(n): # change to list comprehension
            dihedral_indices.append((c_alphas[i], c_alphas[i+1], c_alphas[i+2], c_alphas[i+3]))

        # Compute dihedrals
        dihedrals = compute_dihedrals(ensemble, dihedral_indices)
    
        return dihedrals


    # Accessible surface area (SASA) similarity measure
    def ensemble_sasa(self, ensemble):
        '''Ensemble representation in SASA space
        

        Parameters
        ----------
        ensemble : String
            Filename of an ensemble of protein conformations
        
            
        Returns
        -------
        SASA : np.array, shape=(len(m), len(n))
            The SASA for each amino acid residue n in each 
            conformation m contained in `ensemble`.
            
        '''

        # Load ensembles
        if ensemble[-4:] == '.dcd':
            ensemble = load_dcd(ensemble, self.topology)
        else:
            ensemble = load(ensemble, self.topology) # agnostic load

        # Calculate solvent accessible surface area
        sasa = shrake_rupley(ensemble, mode='residue')
    
        return sasa


    def random_sample(self, arr):
        '''Randomly sample data from the array arr
        

        Parameters
        ----------
        
        arr : np.array, shape=(n, m)
            Output from any of the ensemble_X methods, 
            where X is cbcn, cacn, caba, cata, or sasa
        
            
        Returns
        -------
        sample : np.array, shape=(n, l)
            Random sample from arr with l comformations
            
        '''
        
        m, n = arr.shape
        l=1000 # make a parameter
        idx = np.random.randint(0, m, (l,n))
        
        # randomly select k data points with replacement
        sample = np.take_along_axis(arr, idx, axis=0)

        return sample


    def estimate_pdf(self, arr, x_min, x_max, x_num): # make as parameters
        '''Estimate probability distribution using gaussian kde
        

        Parameters
        ----------
        
        arr : np.array, shape=(n, m)
            Output from any of the ensemble_X methods, 
            where X is cbcn, cacn, caba, cata, or sasa

        x_min : float
              The minimum value of local parameter
        
            
        x_max : float
              The maximum value of local parameter
        
        x_num : int
              The number of descrte points between x_min and x_max
        
	
	Returns
        -------
        probs : np.array, shape=(1, x_num)
            probality densities at x_num descrete points
            
        '''
        

        n = arr.shape[1] # number of local parameters
        #if self.verbose:
        #    print(f'There are {n} values of local structural property.')
        x = np.linspace(x_min, x_max, x_num) 

        probs = np.zeros((len(x), n))

        for i in range(n):
            kde = gaussian_kde(arr[:,i], bw_method='silverman') #make parameter
            probs[:,i]  = kde(x)

        return probs



    def jsd_local(self, ensemble1, ensemble2, x_min, x_max, x_num):
        '''Caclulate the Jensen-Shannon distance between for all local order parameters


        Parameters
        ----------
        
        ensemble1 : np.array, shape=(n, m)
            Output from any of the ensemble_X methods, 
            where X is cbcn, cacn, caba, cata, or sasa

        ensemble2 : np.array, shape=(n, m)
            Output from any of the ensemble_X methods, 
            where X is cbcn, cacn, caba, cata, or sasa
        
        x_min : float
              The minimum value of local parameter
        
            
        x_max : float
              The maximum value of local parameter
        

        x_num : int
              The number of descrte points between x_min and x_max
        

	Returns
        -------
        local_jsds : np.array, shape=(1, x_num)
            The Jensen-Shannon distance at x_num descrete points
            
        '''
    

        # Estimate probability distributions using gaussian kde
        ensemble1_probs = self.estimate_pdf(ensemble1, x_min, x_max, x_num)
        ensemble2_probs = self.estimate_pdf(ensemble2, x_min, x_max, x_num)

        
        # Caclulate jensen-shannon distance metric 
        # (this is the square root of the jensen-shannon divergence)   
        local_jsds = jensenshannon(ensemble1_probs, ensemble2_probs, base=2)
        local_jsds[np.isinf(local_jsds)] = 0
        local_jsds[np.isnan(local_jsds)] = 0


        return local_jsds

    
    def ensemble_representation(self, measure='CBCN'):
        '''Represent protein ensembles using any local measure
        
        Allowed values of measure parameter
        C-alpha contact number (CACN)
        C-beta contact number (CBCN)
        Virtual C-alpha - CA-alpha bond angle (CABA)
        Virtual C-alpha - CA-alpha torsion angle (CATA)
        Solvent accessible surface area (SASA)


        Parameters
        ----------
        
        measure : String
            The local structural property for ensemble representation 
            'CBCN', 'CACN', 'CABA', 'CATA', or 'SASA'


	Returns
        -------
        ensembles : list
            A list of ensemble representations
            
        '''

        # Ensemble representation
 
        ensembles = []

        if self.verbose:
            print(f"There are {len(self.data)} ensemble datasets")

        for i, d in enumerate(self.data):
            if self.verbose:
                print(f'Representing ensemble dataset #{i+1} located at: {d}')

            if measure == 'CACA':
                ens = self.ensemble_caca(d)
           
            elif measure == 'CABA':
                ens = self.ensemble_caba(d)

            elif measure == 'CATA':
                ens = self.ensemble_cata(d)

            elif measure == 'SASA':
                ens = self.ensemble_sasa(d)
                
            elif measure == 'CBCN':
                ens = self.ensemble_cbcn(d)

            elif measure == 'CACN':
                ens = self.ensemble_cacn(d)
        
            else:
                print(f'Error! {measure} is an invalid option for similarity measure.')
                exit() # Terminate this program 

            ensembles.append(ens)

        return ensembles



    def dissimilarity(self, ensemble1, ensemble2, x_min=None, x_max=None, x_num=100, s_num=5):
        '''Calculate dissimilarity between two ensembles


        Parameters
        ----------
        
        ensemble1 : np.array, shape=(n, m)
            Output from any of the ensemble_X methods, 
            where X is cbcn, cacn, caba, cata, or sasa

        ensemble2 : np.array, shape=(n, m)
            Output from any of the ensemble_X methods, 
            where X is cbcn, cacn, caba, cata, or sasa
        
        x_min : float
              The minimum value of local parameter
        
            
        x_max : float
              The maximum value of local parameter
        

        x_num : int
              The number of descrte points between x_min and x_max
        

        s_num : int
              The number of ensemble samples to use in statistical signifance test


	Returns
        -------
        d : tuple, size = 3 
            The dissimilarity with 3 elements: global dissimilarity, local dissimilarity
            and p-values for the local dissimilarity

        '''    

        #initiate x_min and x_max

        # check whether  is none or not
        if x_min == None or x_max == None:
            print(f"x_min or x_max is None!") 
            print(f"Set the x_min and x_max parameters of the ensemble_dissimilarity function.")
            exit()

        #if x_min==0 and x_max==0 and x_num==0:
        #    x_min = np.min([np.min(ensemble1), np.min(ensemble2)]) 
        #    x_max = np.max([np.max(ensemble1), np.max(ensemble2)])
       

        # Statistical significance test
        s = s_num # number of random samples # make a parameter

        samples = {'ensemble1':[], 'ensemble2':[]}

        for i in range(s):
            samples['ensemble1'].append(self.random_sample(ensemble1))
            samples['ensemble2'].append(self.random_sample(ensemble2))

        # Calculate inter ensemble jsds
        inter_ensemble_jsds = []
        for i in range(s):
            for j in range(s):
                #print(f'{i},{j}')
                ens1 = samples['ensemble1'][i]
                ens2 = samples['ensemble2'][j]
                inter_ensemble_jsds.append(self.jsd_local(ens1, ens2, x_min=x_min, x_max=x_max, x_num=x_num))
        inter_jsd = np.stack(inter_ensemble_jsds,axis=0)
                
        # Calculate intra ensemble jsds
        intra_ensemble_jsds = []
        for k in ['ensemble1', 'ensemble2']:
            for i in range(s):
                for j in range(s):
                    if i < j:
                        ens1 = samples[k][i]
                        ens2 = samples[k][j]
                        intra_ensemble_jsds.append(self.jsd_local(ens1, ens2, x_min=x_min, x_max=x_max, x_num=x_num))
        intra_jsd = np.stack(intra_ensemble_jsds,axis=0)

        # Perform Mann-Whitney U rank test
        U1, pvalue = mannwhitneyu(inter_jsd, intra_jsd) # make parameters
        
        # Calculate jensen-shannon divergence (JSD) per coordinate
        local_dissimilarity = self.jsd_local(ensemble1, ensemble2, x_min=x_min, x_max=x_max, x_num=x_num)
        local_dissimilarity[pvalue >= 0.05] = 0.00 # Only JSDs with pvalue < 0.05 is significant

        # Caclulate global dissimilarity as average JSD 
        global_dissimilarity = np.mean(local_dissimilarity)

        d = (global_dissimilarity, local_dissimilarity, pvalue)

        
        return  d

