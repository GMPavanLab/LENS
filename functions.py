import scipy
from scipy.spatial.distance import pdist
from scipy.signal import argrelextrema
from scipy.cluster.hierarchy import complete, fcluster, dendrogram, linkage
from scipy.cluster import hierarchy
import numpy as np
import collections
import MDAnalysis as mda
import MDAnalysis.analysis.rdf as RDF
import matplotlib.pylab as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.cluster import KMeans


def perform_search(XYZ_UNI, COFF, INIT, END, STRIDE, BOX_const=False):

    beads = XYZ_UNI.select_atoms('all')
    list_c = list()
    #loop over traj
    for i,ts in enumerate(XYZ_UNI.trajectory[INIT:END:STRIDE]):
        if BOX_const == True:
            if i==0:
                nsearch = mda.lib.NeighborSearch.AtomNeighborSearch(beads, box=XYZ_UNI.dimensions)
        else:
            nsearch = mda.lib.NeighborSearch.AtomNeighborSearch(beads, box=XYZ_UNI.dimensions)
        cont_list = [nsearch.search(i, COFF, level='A')  for i in beads]          
        list_c.append([[int(j) for j in cont_list[i].ix] for i in range(len(cont_list))])   
    list_sum_noself = dict_count(remove_self(np.sum(list_c, axis=0)))
    X_all = add_zeros(sort_dict(list_sum_noself, 0))
    return X_all


def perform_search_time(XYZ_UNI, COFF, INIT, END, STRIDE, BOX_const=False):
    beads = XYZ_UNI.select_atoms('all')
    cont_list = list()
    #loop over traj
    for i,ts in enumerate(XYZ_UNI.trajectory[INIT:END:STRIDE]):
        if BOX_const == True:
            if i==0:
                nsearch = mda.lib.NeighborSearch.AtomNeighborSearch(beads, box=XYZ_UNI.dimensions)
        else:
            nsearch = mda.lib.NeighborSearch.AtomNeighborSearch(beads, box=XYZ_UNI.dimensions)
        cont_list.append([nsearch.search(i, COFF, level='A')  for i in beads])
    return cont_list


# remove the ith count from the ith bead contact list 
def remove_self(v):
    v_rm = list()
    for i,x in enumerate(v):
        v_rm.append(list(filter(lambda j: j != i, x)))
    return v_rm

# perform the count
def dict_count(v):
    v_count = list()
    for i,N in enumerate(v):
        v_count.append(dict(collections.Counter(N)))
    return v_count

# sort the {bead, count}
def sort_dict(v, key):
    v_sorted = list()
    for c in v:
        v_sorted.append(dict(sorted(c.items(), key=lambda item: item[key])))
    return v_sorted

# add zeros when there is never contact
def add_zeros(X):
    N=len(X)
    X_all = list()
    for x in X:
        x_all = list()
        count = 0
        for atm in x:
            if atm != count:
                while count < atm:
                    x_all.append([count, 0])
                    count+=1
            if atm == count:
                x_all.append([atm, x[atm]])
            count+=1
        while count < N:
            x_all.append([count, 0])
            count+=1

        X_all.append(x_all)
    return X_all


def clusters_local(v,dynamic):
    cl_tot = list()
    particle = [i for i in range(np.shape(v)[0])]
    for p in particle:
        cl = list()
        for f_desc in v[p]:
            if f_desc > dynamic:
                cl.append(1)
            else:
                cl.append(0)
        cl_tot.append(cl)
    return cl_tot

def flatten(vector):
    trans = np.transpose(vector)
    trans_fl = np.reshape(trans, np.shape(trans)[0]*np.shape(trans)[1])
    return trans_fl


def xyz_to_uni(xyz, box=None, traj_name_output=None):
    
    # reading input xyz and box dimensions
    xyz_u = mda.Universe(xyz)
    n_monomers = len(xyz_u.atoms)
    n_frames = len(xyz_u.trajectory)

    if not isinstance(box, str):
        xyz_box_sigle = mda.lib.mdamath.triclinic_box(box[0],box[1],box[2])[:3]
        xyz_box = np.array([xyz_box_sigle for i in range(n_frames)])
        angles = list(mda.lib.mdamath.triclinic_box(box[0],box[1],box[2])[3:])
    else:
        xyz_box = np.loadtxt(box)
        angles = [90.,90.,90.]
        
    # new Universe init
    new_u = mda.Universe.empty(n_atoms=n_monomers,
                               n_residues=n_monomers,
                               n_segments=1,
                               atom_resindex=np.arange(n_monomers),
                               residue_segindex=[1]*n_monomers,
                               trajectory=True)
    
    # init new empty traj
    new_traj = np.empty((n_frames, n_monomers, 3))
    
    # loop over empty frame to insert the new frames
    for i,ts in enumerate(xyz_u.trajectory):
        empty_frame = np.empty((n_monomers, 3))
        for j,pos in enumerate(xyz_u.atoms.positions):
            empty_frame[j] = pos
        new_traj[i] = empty_frame
        
    # add the dimension of the box to the new traj
    new_u.load_new(new_traj, format=mda.coordinates.memory.MemoryReader)
    for s,snap in enumerate(new_u.trajectory):
        box_dim_tmp = np.pad(xyz_box[s], (0, 3), 'constant') + np.array([0.,0.,0.]+angles)
        new_u.trajectory[snap.frame].dimensions = box_dim_tmp
        
    return new_u

    
def get_gofr_all(universe,b, e, s,rcut, bins):
  
    selection = universe.atoms
    rdf_ = RDF.InterRDF(selection, selection, nbins=bins,range=(0,rcut), exclusion_block=(1,1))
    rdf_.run(start=b, stop=e, step=s)
    rdf_file_shaped_raw = [[rdf_.bins[i], rdf_.rdf[i]] for i in range(len(rdf_.bins))]
    rdf_file = np.vstack(rdf_file_shaped_raw)
    
    return rdf_,rdf_file

def write_stat_clusters(cl_sort, outfile, INIT, END, STRIDE):
    with open(outfile, 'w+') as file:
        for frame in range(INIT, END, STRIDE):
            for i in cl_sort:
                file.write(str(i[1]-1)+'\n')

def dynamicity(clusters, M):
    
    cl = [ [i, clusters[i]] for i in range(len(clusters))]
    dyn = [ [i, 1./np.std(M[i])] for i in range(len(M))]

    cl_sort = sorted(cl, key=lambda x: x[0])
    dyn_sort = sorted(dyn, key=lambda x: x[0])

    cl_dyn  = [np.transpose(cl_sort)[1], np.transpose(dyn_sort)[1]] 
    cl_dyn_T = np.transpose(cl_dyn)
    cl_dyn_T = cl_dyn_T[cl_dyn_T[:, 0].argsort()]
    groups_dyn = np.split(cl_dyn_T[:,1], np.unique(cl_dyn_T[:, 0], return_index=True)[1][1:])
    
    return cl_dyn, groups_dyn

def local_dynamics(list_sum):
    particle = [i for i in range(np.shape(list_sum)[1])]
    ncont_tot = list()
    nn_tot = list()
    num_tot = list()
    den_tot  = list()
    for p in particle:
        ncont = list()
        nn = list()
        num = list()
        den = list()
        for frame in range(len(list_sum)):
            if frame == 0:
                ncont.append(0)
                nn.append(0)
            else:
                # se il set di primi vicini cambia totalmente, l'intersezione è lunga 1 ovvero la bead self
                # vale anche se il numero di primi vicini prima e dopo cambia
                if len(list(set(list_sum[frame-1][p]) & set(list_sum[frame][p])))==1:
                    # se non ho NN lens è 0
                    if len(list(set(list_sum[frame-1][p])))==1 and len(set(list_sum[frame][p]))==1:
                        ncont.append(0)
                        nn.append(0)
                        num.append(0)
                        den.append(0)
                    # se ho NN lo metto 1
                    else:
                        ncont.append(1)
                        nn.append(len(list_sum[frame][p])-1)
                        num.append(1)
                        den.append(len(list_sum[frame-1][p])-1+len(list_sum[frame][p])-1)    
                else:
                    # contrario dell'intersezione fra vicini al frame f-1 e al frame f
                    c_diff = set(list_sum[frame-1][p]).symmetric_difference(set(list_sum[frame][p]))
                    ncont.append(len(c_diff)/(len(list_sum[frame-1][p])-1+len(list_sum[frame][p])-1))
                    nn.append(len(list_sum[frame][p])-1)
                    num.append(len(c_diff))
                    den.append(len(list_sum[frame-1][p])-1+len(list_sum[frame][p])-1)
        num_tot.append(num)
        den_tot.append(den)
        ncont_tot.append(ncont)
        nn_tot.append(nn)
    return ncont_tot, nn_tot, num_tot, den_tot


def check(value, b_chunk):
    if b_chunk[0] <= value < b_chunk[1]:
        return True
    return False


def PLTmatrixrates_label_colors_perc(data, colors, vmax, axes, fontsize=23):
    ax = axes
    
    sns.heatmap(data*100, fmt='.1f', vmin=0, vmax=vmax, linewidths=0.5, linecolor='white', cmap='Greys',  annot=True, alpha=1., annot_kws={"fontsize":fontsize},  square=True, cbar=False, ax=ax)

    TICKYPOS = -0.3
    tickslabels = [i+0.5 for i in range(len(data))]

    for i in range(len(data)):
        ax.add_patch(patches.Circle((tickslabels[i],TICKYPOS), facecolor=colors[i], lw=1.5, edgecolor='black', radius=.2,
                           clip_on=False))
        ax.add_patch(patches.Circle((TICKYPOS,tickslabels[i]), facecolor=colors[i], lw=1.5, edgecolor='black', radius=.2,
                           clip_on=False))
        
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    ax.get_xaxis().set_ticklabels([])
    ax.xaxis.tick_top()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax

def savgol_filter(ncont_tot,polyorder,window,plot=True, ylim=None, xticks=None, xticks_l=None,yticks=None, yticks_l=None, xunit='$\mu$', windows_study=[10,50,100,150],polyorder_study=[2,4,6]):
    ncont_rolling = list()
    particle = [i for i in range(np.shape(ncont_tot)[0])]
    for p in particle:
        savgol_2_10 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[0],polyorder=polyorder_study[0])
        savgol_2_50 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[1],polyorder=polyorder_study[0])
        savgol_2_100 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[2],polyorder=polyorder_study[0])
        savgol_2_150 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[3],polyorder=polyorder_study[0])

        savgol_4_10 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[0],polyorder=polyorder_study[1])
        savgol_4_50 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[1],polyorder=polyorder_study[1])
        savgol_4_100 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[2],polyorder=polyorder_study[1])
        savgol_4_150 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[3],polyorder=polyorder_study[1])


        savgol_6_10 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[0],polyorder=polyorder_study[2])
        savgol_6_50 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[1],polyorder=polyorder_study[2])
        savgol_6_100 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[2],polyorder=polyorder_study[2])    
        savgol_6_150 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[3],polyorder=polyorder_study[2])

        savgol = scipy.signal.savgol_filter(ncont_tot[p], window_length=window,polyorder=polyorder)
        ncont_rolling.append(savgol[int(window/2):-int(window/2)])
        
        if p%100==0 and plot:
            fig, ax = plt.subplots(1,4, figsize=(16,4), dpi=500)
            fig.suptitle(r'Bead ID '+str(p), size=20)
            ax[0].set_title("window: " +str(windows_study[0]), fontsize=20)
            ax[0].plot(savgol_2_10, c='green', label='poly order '+str(polyorder_study[0]))
            ax[0].plot(savgol_4_10, c='blue', label='poly order '+str(polyorder_study[1]))
            ax[0].plot(savgol_6_10, c='red', label='poly order '+str(polyorder_study[2]))
            ax[0].plot(ncont_tot[p], c='gray',lw=0.5, alpha=0.8)
            ax[0].set_ylim(ylim)
            ax[0].set_ylabel(r'$\delta_b^{\tau}$', size=20)
            ax[0].set_xlabel(r't ['+xunit+'s]', size=20)
            ax[0].set_xticks(xticks, fontsize=20)
            ax[0].set_xticklabels(xticks_l, fontsize=20)
            ax[0].set_yticks(yticks, fontsize=20)
            ax[0].set_yticklabels(yticks_l, fontsize=20)
            ax[0].legend()

            ax[1].set_title("window: " +str(windows_study[1]), fontsize=20)
            ax[1].plot(savgol_2_50, c='green', label='poly order '+str(polyorder_study[0]))
            ax[1].plot(savgol_4_50, c='blue', label='poly order '+str(polyorder_study[1]))
            ax[1].plot(savgol_6_50, c='red', label='poly order '+str(polyorder_study[2]))
            ax[1].plot(ncont_tot[p], c='gray',lw=0.1, alpha=0.8)
            ax[1].set_ylim(ylim)
            ax[1].set_xlabel(r't ['+xunit+'s]', size=20)
            ax[1].set_xticks(xticks, fontsize=20)
            ax[1].set_xticklabels(xticks_l, fontsize=20)
            ax[1].set_yticks([], fontsize=20)
            ax[1].set_yticklabels([], fontsize=20)
            ax[1].legend()

            ax[2].set_title("window: " +str(windows_study[2]), fontsize=20)        
            ax[2].plot(savgol_2_100, c='green', label='poly order '+str(polyorder_study[0]))
            ax[2].plot(savgol_4_100, c='blue',label='poly order '+str(polyorder_study[1]))
            ax[2].plot(savgol_6_100, c='red', label='poly order '+str(polyorder_study[2]))
            ax[2].plot(ncont_tot[p], c='gray',lw=0.1, alpha=0.8)
            ax[2].set_ylim(ylim)
            ax[2].set_xlabel(r't ['+xunit+'s]', size=20)
            ax[2].set_xticks(xticks, fontsize=20)
            ax[2].set_xticklabels(xticks_l, fontsize=20)
            ax[2].set_yticks([], fontsize=20)
            ax[2].set_yticklabels([], fontsize=20)
            ax[2].legend()

            ax[3].set_title("window: " +str(windows_study[3]), fontsize=20)
            ax[3].plot(savgol_2_150, c='green', label='poly order '+str(polyorder_study[0]))
            ax[3].plot(savgol_4_150, c='blue', label='poly order '+str(polyorder_study[1]))
            ax[3].plot(savgol_6_150, c='red', label='poly order '+str(polyorder_study[2]))
            ax[3].plot(ncont_tot[p], c='gray',lw=0.1, alpha=0.8)
            ax[3].set_ylim(ylim)
            ax[3].set_xlabel(r't ['+xunit+'s]', size=20)
            ax[3].set_xticks(xticks, fontsize=20)
            ax[3].set_xticklabels(xticks_l, fontsize=20)
            ax[3].set_yticks([], fontsize=20)
            ax[3].set_yticklabels([], fontsize=20)        
            ax[3].legend()   
            plt.tight_layout()
    return ncont_rolling