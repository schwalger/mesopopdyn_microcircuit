import numpy as np

#########################################################################
# BEGIN EXCERPT OF network_params.py from Potjans2014, downloaded from github/pynn/examples
########################################################################


###################################################
###     	Network parameters		###        
###################################################

# Relative inhibitory synaptic weight
g = -4.

neuron_params = {
    'cm'        : 0.25,  # nF
    'i_offset'  : 0.0,   # nA
    'tau_m'     : 10.0,  # ms
    'tau_refrac': 2.0,   # ms
    'tau_syn_E' : 0.5,   # ms
    'tau_syn_I' : 0.5,   # ms
    'v_reset'   : -65.0, # mV
    'v_rest'    : -65.0, # mV
    'v_thresh'  : -50.0  # mV
}

layers = {'L23':0, 'L4':1, 'L5':2, 'L6':3}
n_layers = len(layers)
pops = {'E':0, 'I':1}
n_pops_per_layer = len(pops)
structure = {'L23': {'E':0, 'I':1},
             'L4' : {'E':2, 'I':3},
             'L5' : {'E':4, 'I':5},
             'L6' : {'E':6, 'I':7}}

# Numbers of neurons in full-scale model
N_full = {
  'L23': {'E':20683, 'I':5834},
  'L4' : {'E':21915, 'I':5479},
  'L5' : {'E':4850,  'I':1065},
  'L6' : {'E':14395, 'I':2948}
}

establish_connections = True

# Probabilities for >=1 connection between neurons in the given populations. 
# The first index is for the target population; the second for the source population
#             2/3e      2/3i    4e      4i      5e      5i      6e      6i
conn_probs = [[0.1009,  0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.    ],
             [0.1346,   0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.    ],
             [0.0077,   0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.    ],
             [0.0691,   0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.    ],
             [0.1004,   0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
             [0.0548,   0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.    ],
             [0.0156,   0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364,   0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]]

# In-degrees for external inputs
K_ext = {
  'L23': {'E':1600, 'I':1500},
  'L4' : {'E':2100, 'I':1900},
  'L5' : {'E':2000, 'I':1900},
  'L6' : {'E':2900, 'I':2100}
}

# Mean rates in the full-scale model, necessary for scaling
# Precise values differ somewhat between network realizations
full_mean_rates = {
  'L23': {'E':0.971, 'I':2.868},
  'L4' : {'E':4.746, 'I':5.396},
  'L5' : {'E':8.142, 'I':9.078},
  'L6' : {'E':0.991, 'I':7.523}
}

# Mean and standard deviation of initial membrane potential distribution
V0_mean = -58. # mV
V0_sd = 5.     # mV

# Background rate per synapse
bg_rate = 8. # spikes/s

# Mean synaptic weight for all excitatory projections except L4e->L2/3e
w_mean = 87.8e-3 # nA
w_ext = 87.8e-3 # nA
# Mean synaptic weight for L4e->L2/3e connections 
# See p. 801 of the paper, second paragraph under 'Model Parameterization', 
# and the caption to Supplementary Fig. 7
w_234 = 2*w_mean # nA

# Standard deviation of weight distribution relative to mean for 
# all projections except L4e->L2/3e
w_rel = 0.1
# Standard deviation of weight distribution relative to mean for L4e->L2/3e
# This value is not mentioned in the paper, but is chosen to match the 
# original code by Tobias Potjans
w_rel_234 = 0.05

# Means and standard deviations of delays from given source populations (ms)
d_mean = {'E': 1.5, 'I': 0.75}
d_sd = {'E': 0.75, 'I': 0.375}

# Parameters for transient thalamic input
thalamic_input = False
thal_params = {
  # Number of neurons in thalamic population
  'n_thal'      : 902,
  # Connection probabilities
  'C'           : {'L23': {'E': 0, 'I':0},
                   'L4' : {'E':0.0983, 'I':0.0619},
                   'L5' : {'E': 0, 'I':0},
                   'L6' : {'E':0.0512, 'I':0.0196}},
  'rate'        : 120., # spikes/s;
  'start'       : 700., # ms
  'duration'    : 10.   # ms;
}

#########################################################################
# END EXCERPT OF network_params.py from Potjans2014, downloaded from github/pynn/examples
########################################################################


##########################################################################
## BEGIN EXCERPT OF network.py from Potjans2014, downloaded from github/pynn/examples
#########################################################################


def create_weight_matrix():
    w = np.zeros([n_layers*n_pops_per_layer, n_layers*n_pops_per_layer])
    for target_layer in layers:
        for target_pop in pops:
            target_index = structure[target_layer][target_pop]
            for source_layer in layers:
                for source_pop in pops:
                    source_index = structure[source_layer][source_pop]
                    if source_pop == 'E':
                        if source_layer == 'L4' and target_layer == 'L23' and target_pop == 'E':
                            w[target_index][source_index] = w_234
                        else:
                            w[target_index][source_index] = w_mean
                    else:
                        w[target_index][source_index] = g*w_mean
    return w



##########################################################################
## END EXCERPT OF network.py from Potjans2014, downloaded from github/pynn/examples
#########################################################################


weight_matrix = create_weight_matrix()


def create_thalamic_drive_vector():
    w = np.zeros( n_layers*n_pops_per_layer )
    for target_layer in layers:
        for target_pop in pops:
            target_index = structure[target_layer][target_pop]
            w[target_index] = thal_params['C'][target_layer][target_pop] * \
                thal_params['n_thal'] * thal_params['rate']
    return w
    
tha_drive_vec = create_thalamic_drive_vector()


# 2/3e      2/3i    4e      4i      5e      5i      6e      6i
def get_mu_DeltaV():
    K_ext_vec = np.array([ K_ext['L23']['E'], K_ext['L23']['I'],\
                           K_ext['L4']['E'], K_ext['L4']['I'],\
                           K_ext['L5']['E'], K_ext['L5']['I'],\
                           K_ext['L6']['E'], K_ext['L6']['I'],\
                          ])
    rate_ext_vec = K_ext_vec * bg_rate
    taum = neuron_params['tau_m'] * 1e-3
    taus = neuron_params['tau_syn_E'] * 1e-3
    mu = w_ext * taus * rate_ext_vec * taum / neuron_params['cm'] * 1e3
    V_free_std = (taum * taus * w_ext / neuron_params['cm']) * np.sqrt(0.5 * rate_ext_vec / (taum+taus)) * 1e3
    print(mu)
    print(V_free_std)
    DeltaV = 1.5 * V_free_std
    return mu, DeltaV



