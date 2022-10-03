#%%
import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy.fft import rfft, rfftfreq
import openseespy.opensees as ops

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from IPython.display import HTML

# --------------------------------------------------------------- #
def get_random_figures(K, bound, seed):
    """ Generate K figures in the range of bound.
    """
    rng = default_rng(seed)
    a, b = bound
    return np.floor(rng.uniform(a, b, K) * 10000) / 10000
    
def get_damped_signal(K, T=3, fs=100, 
                      bound=((2, 5), (0.01, 0.02), (3, 10), (0, 3.14)), 
                      noise_std=0, seed=123):
    """ Get a multi-component damped signal.
    
    $$ y_i = a * exp(-2*pi*f*zeta*t) * cos(2*pi*f*t + phi) $$
    
    Parameters
    ----------
    `K`: int
        number of components
    `T`: float
        duration, defaults to 3
    `fs`: float
        sampling rate, defaults to 100
    `bound`: tuple
        ((initialAmplitude_lower, initialAmplitude_upper), (dampingRatio_lower,
        dampintRatio_upper), (frequency_lower, frequency_upper)), (initialPhase
        _lower, initialPhase_upper)), frequency in Hz, phase in rad
    `noise_std`:
        std of noise, defaults to 0
    `seed`: int
        seed of generating pesudo rand numbers, defaults to 123

    Returns
    -------
    `out` : dict
        collections of the generated signal 
    """
    # time
    t = np.linspace(0, T, T*fs, endpoint=False)
    
    # initial amplitude
    amp = get_random_figures(K, bound[0], seed)
    # damping ratio
    damp_r = get_random_figures(K, bound[1], seed+1)
    # frequency
    freq = get_random_figures(K, bound[2], seed+2)
    freq = np.sort(freq) * 2 * np.pi
    # phase
    phase = get_random_figures(K, bound[3], seed+3)
    
    # components
    Y = np.zeros((len(t), K))
    for k in range(K):
        Y[:, k] = amp[k] * np.exp(-freq[k] * damp_r[k] * t) *\
            np.cos(freq[k] * t + phase[k])
            
    # noise
    rng = default_rng(seed+4)
    noise = rng.normal(0, noise_std, len(t)) # white noise, ~ N(0, noise_std)
    
    # signal with noise
    y = np.sum(Y, axis=1) + noise
    
    return {
        'y': y,             # total signal
        'noise': noise,     # noise
        'Y': Y,             # components
        'time': t,          # time series
        'fs': fs,           # sampling rate
        'amp': amp,           # initial amplitude
        'damp_r': damp_r,     # damping ratio
        'freq': freq/2/np.pi,          # frequency
        'phase': phase        # initial phase
    }

#------------------------------------------------------------
def get_response_from_simply_supported_beam(
        struct_params={"L_total": 10, "num_node": 41, "E_mod": 2.05e11, 
                       "Area": 14.3e-4, "Iz": 245e-8, "mass": 7850*14.3e-4},
        analysis_params={"zeta": 0.02, "fs": 500, "tol_time": 5, 
                         "num_eig": 5, "idx_for_damp_ratio": [0, 2]},
        load_params={"pattern": 0, "duration": 0.5, "magnitude": 1000,
                     "location": [6], "seed": 111}):
    """Get response from a simply suppored beam.
    
    ## Cases included:
        - Free response
        - Excited response by random force

    Parameters:
    ----------
        struct_params: _description_
        analysis_params: _description_
        load_params: _description_

    Returns:
    -------
        
    """
    # remove existing model
    ops.wipe()

    # set modelbuilder
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # create nodes
    L_total = struct_params['L_total']
    num_node = struct_params['num_node']
    L0 = L_total / (num_node - 1)
    for i in range(num_node):
        ops.node(i+1, i*L0, 0.0)

    # assign geometric transformation
    BeamTransfTag = 1
    ops.geomTransf('Linear', BeamTransfTag)

    # define elements
    # element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, Iz,
    # transfTag, <'-mass', mass>, <'-cMass'>, <'-release', releaseCode>)
    E_mod = struct_params['E_mod']
    Area = struct_params['Area']
    Iz = struct_params['Iz']
    mass = struct_params['mass']
    for i in range(1, num_node):
        ops.element('elasticBeamColumn', i, i, i+1, Area,
                    E_mod, Iz, BeamTransfTag, '-mass', mass)

    # set boundary condition
    ops.fix(1, 1, 1, 0)
    ops.fix(num_node, 0, 1, 0)

    # modal analysis
    num_eig = analysis_params['num_eig']
    eig_val = ops.eigen(num_eig)
    Phi = np.zeros((num_node, num_eig))
    dof = 2  # y-direction
    for i in range(num_eig):
        for node_tag in range(num_node):
            Phi[node_tag, i] = ops.nodeEigenvector(node_tag+1, i+1, dof)
        
    # rayleigh(alphaM, betaK, betaKinit, betaKcomm)
    zeta = analysis_params['zeta']
    i0, i1 = analysis_params['idx_for_damp_ratio']
    omega = np.sqrt(eig_val) # unit: rand/s
    alphaM = 2 * zeta / (omega[i0] + omega[i1]) * omega[i0] * omega[i1]
    betaK = 2 * zeta / (omega[i0] + omega[i1])
    # ops.region(1, '-ele', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    # '-rayleigh', alphaM, betaK, 0.0, 0.0)
    ops.rayleigh(alphaM, betaK, 0, 0)

    # load pattern
    ts_rect = 1
    pattern_rect = 1
    ts_random = 2
    pattern_random = 2
    dir = 2  # y-direction
    t_start = 0
    t_end = load_params['duration']
    
    pattern = load_params['pattern']
    mag = load_params['magnitude']
    loc = load_params['location']
    seed = load_params['seed']

    fs = analysis_params['fs']              # sampling rate
    dt_analysis = 1 / fs		            # timestep of analysis
    tol_time = analysis_params['tol_time']  # total time of analysis
    
    # burst force on selected nodes
    if pattern == 0: 
        ops.timeSeries('Rectangular', ts_rect, t_start, t_end, '-factor', 1)
        ops.pattern('Plain', pattern_rect,  ts_rect)
        #        nd  FX, FY, MZ
        [ops.load(ni, 0., mag, 0.) for ni in loc]
    # random force on selected nodes
    else: 
        for ni in loc:
            rng = default_rng(seed+ni)
            random_force = rng.normal(0, 1, fs*tol_time).tolist()
            force_time = np.arange(0, fs*tol_time, dt_analysis).tolist()
            ops.timeSeries('Path', ts_random+ni, '-dt', dt_analysis, '-values',
                        *random_force, '-time', *force_time)
            ops.pattern('Plain', pattern_random+ni, ts_random+ni)
            ops.load(ni, 0., mag, 0.)

    # define dynamic analysis parameters
    ops.wipeAnalysis()			            # destroy all components of the 
                                            # Analysis object
    ops.constraints('Plain')		        # how it handles boundary conditions
    ops.numberer('RCM')				        # renumber dof's to minimize band-width (optimization)
    ops.system('BandGen')			        # how to store and solve the system of equations in the analysis
    ops.test('NormDispIncr', 1.0e-8, 100)   # type of convergence criteria with tolerance, max iterations
    ops.algorithm('NewtonLineSearch')       # use NewtonLineSearch solution algorithm
    ops.integrator('Newmark', 0.5, 0.25)    # uses Newmark's average acceleration method to compute the time history
    ops.analysis('Transient')			    # type of analysis: transient or static
    num_steps = tol_time * fs               # number of steps in analysis

    # perform the dynamic analysis and collect results
    ops.reset()
    
    time = np.zeros(num_steps)
    node_disp = np.zeros((num_steps, num_node))
    node_vel = np.zeros((num_steps, num_node))
    node_acc = np.zeros((num_steps, num_node))


    for i in range(num_steps):
        ops.analyze(1, dt_analysis)
        time[i] = ops.getTime()
        # collect disp for element nodes
        for j in range(num_node):
            node_disp[i, j] = ops.nodeDisp(j+1, dir)
            node_vel[i, j] = ops.nodeVel(j+1, dir)
            node_acc[i, j] = ops.nodeAccel(j+1, dir)
            
    # theory frequency
    freq_theory = [(((i+1)*np.pi)/L_total)**2 * np.sqrt(E_mod*Iz/mass) /
                  (2*np.pi) for i in range(num_eig)]
    
    # damping ratio
    zeta1 = np.zeros(num_eig)
    for i in range(num_eig):
        zeta1[i] = 1 / 2 * (alphaM / omega[i] + omega[i] * betaK)
    
    return {
        "disp": node_disp,
        "vel": node_vel,
        "acc": node_acc,
        "time": time,
        "freq": np.array(omega/2/np.pi),   # unit: Hz
        "freq_theory": np.array(freq_theory), # unit: Hz
        "damp_r": zeta1,
        "mode": Phi
    }


# =============================================================== #
# ======================== plotting ============================= #
# =============================================================== #
def plot_damped_signal(figsize, signal):
    t = signal['time']
    y = signal['y']
    Y = signal['Y']
    noise = signal['noise']
    fs = signal['fs']
    
    fig = plt.figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(t, y, color='#ee0044')
    ax.plot(t, noise, color='#0f0f0f')
    ax.legend(['y', 'noise'])
    ax.set_xlabel('Time/s')

    ax = fig.add_subplot(1, 2, 2)
    n = len(t)
    Yf = rfft(y) * 2 / n
    xf = rfftfreq(n, 1/fs)
    ax.plot(xf, np.abs(Yf), '.-', markersize=2)
    ax.set_xlabel('Frequency/Hz')
    plt.show(block=True)
    return fig

# --------------------------------------------------------------- #
def plot_mode(figsize, mode):
    # print mode
    m, n = mode.shape
    loc = np.arange(0, m)
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.plot(loc, mode, 'o-', markersize=3)
    ax.plot(loc, np.zeros(loc.shape), ls='--', c='#454545')
    ax.legend([f'mode {i+1}' for i in range(n)])
    plt.show(block=True)
    return fig


def plot_response(figsize, time, sig, sensor_locs, fs, t_start=0):
    fig = plt.figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    for j, s in enumerate(sensor_locs):
        a = int(t_start*fs)
        ax.plot(time[a:], sig[a:, j], label=f'node: {s:1d}')
    ax.grid()
    ax.legend()
    ax.set_ylabel('Node acc.')
    ax.set_xlabel('Time/s')

    ax = fig.add_subplot(1, 2, 2)
    n = sig[a:, :].shape[0]
    for j, s in enumerate(sensor_locs):
        Yf = rfft(sig[a:, j]) * 2 / n
        xf = rfftfreq(n, 1/fs)
        ax.semilogy(xf, np.abs(Yf), '.-', label=f'node: {s:1d}', markersize=2)
    ax.set_ylabel('Spectrum of node acc.')
    ax.set_xlabel('Frequency/Hz')
    plt.show(block=True)
    return fig


def vibration_movie(figsize, data):
    N = data.shape[1]
    locs = np.linspace(1, N, N, endpoint=True)
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.plot(locs, np.zeros_like(locs), '--', color='#484848')
    ln, = ax.plot(locs, data[0], '-o', markersize=3, color='#CC0055')
    tx = ax.text(N/4, 0.001, f'{0:.3f}s')
    ax.set_ylim(-0.1, 0.1)

    def animate(i):
        ln.set_ydata(data[i])
        m = np.max(np.abs(data[i]))
        # ax.set_ylim(-1.1*m, 1.1*m)
        tx.set_text(f'{i/200:.3f}s')
        return ln, tx

    anim = animation.FuncAnimation(fig, animate, np.arange(1, data.shape[0]),
                                  blit=True, interval=5)
    writergif = animation.PillowWriter(fps=50, bitrate=100)
    anim.save('../figures/VR.gif', writer=writergif)
    # HTML(anim.to_html5_video()) 
    plt.show()
    return anim
           

#%%
if __name__ == '__main__':
    # ---------------------------------------------------------- #
    sig = get_damped_signal(3, T=3, fs=100, noise_std=0, seed=123)
    plot_damped_signal((12, 4),sig)

    # ---------------------------------------------------------- #
    analysis_params = {
        "zeta":                 0.02,
        "fs":                   500,
        "tol_time":             1,
        "num_eig":              5,
        "idx_for_damp_ratio":   [0, 2]
    }
    # pattern
    # 0: burst force
    # 1: random force on single node
    # 2: random force on  5 nodes
    load_params = {"pattern": 0, "duration": 0.5, "magnitude": 1000,
                   "location": [6], "seed": 111}     

    sensor_locs = [7, 14, 20, 26, 33]

    res0 = get_response_from_simply_supported_beam(
        analysis_params=analysis_params, load_params=load_params)
    
    plot_mode((8, 4), res0['mode'])
    
    res_FEM = pd.DataFrame({
        # "Theoretical freqency": res0['freq_theory'],
        "Freqency": res0['freq'],
        "Damping ratio": res0['damp_r'],
        "Damping factor": 2*np.pi*res0['freq']*res0['damp_r']
    })
    print(res_FEM)
    plot_response((12, 4), res0['time'], res0['acc'][:, sensor_locs], 
                  sensor_locs, analysis_params['fs'], t_start=0.51)
    # vibration_movie((12, 4), res0['disp'])

    load_params = {"pattern": 1, "duration": 0.5, "magnitude": 1000,
               "location": [10], "seed": 111}
    res1 = get_response_from_simply_supported_beam(
        analysis_params=analysis_params, load_params=load_params)
    plot_response((12, 4), res1['time'], res1['acc'][:, sensor_locs],
                  sensor_locs, analysis_params['fs'])

# %%
