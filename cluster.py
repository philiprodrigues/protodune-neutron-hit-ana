import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from itertools import cycle, islice
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import argparse
import arrow
import os.path

def find_time_limits(files):
    """Get the latest minimum and earliest maximum hit time in `files`, ie, the time span that's been read out in all the files"""
    tmin=0
    tmax=2**60
    for f in files:
        arr=np.load(f, mmap_mode="r")[:,1]
        tmin=max(tmin, np.min(arr))
        tmax=min(tmax, np.max(arr))
    return tmin,tmax

def munged_hits_for_times_one_file(filename, tmin, tmax):
    fake_hits=[]
    # mmap_mode=c is copy-on-write: assignments affect data in
    # memory, but changes are not saved to disk.  The file on disk
    # is read-only.
    tmp=np.load(filename,mmap_mode="c")
    ts=tmp[:,1]
    tselector=np.logical_and(ts>tmin, ts<tmax)
    # Select just the collection channels
    chs=tmp[:,0]

    coll_inds=chs%2560>=1600
    selector=np.logical_and(coll_inds, tselector)
    chs_coll=chs[selector]
    coll_hits=tmp[selector]

    if coll_hits.size==0:
        print("Warning: no collection hits selected in {} for {}-{}".format(filename, tmin, tmax))
        return coll_hits
    
    # For high-angle tracks, each hit is long (in time), and its
    # start time is separated from the next hit by a larger
    # distance than for a low-angle tracks. So DBSCAN sees the
    # points widely separated and doesn't turn the track into a
    # cluster. To try to work around this: For every
    # `fake_hit_factor` ticks that the hit is over threshold,
    # create a new hit, in the hope of making high-angle tracks
    # "reconstruct" better.
    fake_hit_factor=10
    tovers=coll_hits[:,3]//(25*fake_hit_factor) # in 2MHz TPC ticks, hence the 25
    coll_hits=np.repeat(coll_hits, tovers, axis=0) # Repeat each hit `tovers` times
    # Add 25 to the time of each subsequent fake hit in a real hit
    hit_diffs=np.hstack([np.arange(0,i*25*fake_hit_factor,25*fake_hit_factor) for i in tovers])
    coll_hits[:,1]+=hit_diffs

    return coll_hits

def contiguified_channels(chs):
    """
    Given the channel numbers of the hits that exist, find the channel
    numbers in [min channel, max channel] that *don't* occur. Presumably
    those are noisy channels that were excluded from the hit finding,
    which result in gaps in the tracks here. So renumber the channels in
    such a way that the noisy channels don't get a number, and the "live"
    channel numbers are contiguous
    """
    orig=chs.copy()-np.min(chs)
    counts=np.bincount(orig)
    #print(np.argwhere(counts==0))
    cumul=np.cumsum(np.where(counts==0, 0, 1))
    return np.array([cumul[i] for i in orig])

def munged_hits_for_times(files, tmin, tmax):
    all_hits=None
    for f in files:
        coll_hits=munged_hits_for_times_one_file(f, tmin, tmax)
        if coll_hits.size==0: continue
        if all_hits is None:
            all_hits=coll_hits[:,0:2]
        else:
            all_hits=np.vstack((all_hits, coll_hits[:,0:2]))

    chs=contiguified_channels(all_hits[:,0])
    all_hits[:,0]=chs
    # Need to rescale the values so they're in the same units, ie, length. Collection channel pitch is 5mm; drift speed is ~1mm/us

    ts=all_hits[:,1]
    chs*=5
    ts//=50 # Integer division. Not perfect, but probably good enough
    ts-=np.min(ts) # Make the absolute values a bit nicer

    return all_hits

def uniquify_filename(filename):
    basename,ext=filename.split(".")
    counter=0
    while os.path.exists(filename):
        filename="{}-{}.{}".format(basename,counter,ext)
        counter+=1
    return filename

def find_clusters(files, tmin, tmax, plot=False, plot_label=None):
    all_hits=munged_hits_for_times(files, tmin, tmax)
    # This code copied from
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    db = DBSCAN(eps=20, min_samples=5).fit(all_hits)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_cluster_hits = len(all_hits) - n_noise
    print("t=[{}, {}]: Found {} clusters, {} track hits, {} noise hits".format(tmin, tmax, n_clusters, n_cluster_hits, n_noise))
    
    # print('Estimated number of clusters: %d' % n_clusters)
    # print('Estimated number of noise points: %d' % n_noise)

    if plot:
        # Code for colouring the points by cluster copied from
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

        fig,ax=plt.subplots()
        plt.scatter(all_hits[:, 1], all_hits[:, 0], color="black", rasterized=True)
        ax.set_xlabel("Time $\\times$ drift velocity (mm)")
        ax.set_ylabel("Pseudo-channel")
        plt.tight_layout()
        fig.savefig(uniquify_filename("all-hits-evt-disp-%s.pdf" % plot_label))
        plt.close(fig)
        
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(labels) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])

        fig2,ax2=plt.subplots()
        plt.scatter(all_hits[:, 1], all_hits[:, 0], color=colors[labels], rasterized=True)
        ax2.set_xlabel("Time $\\times$ drift velocity (mm)")
        ax2.set_ylabel("Pseudo-channel")
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())
        plt.tight_layout()
        fig2.savefig(uniquify_filename("all-hits-clustered-evt-disp-%s.pdf" % plot_label))
        plt.close(fig2)
        
        # Show just the noise hits
        fig3,ax3=plt.subplots()
        isnoise=labels==-1
        plt.scatter(all_hits[:,1][isnoise], all_hits[:,0][isnoise], color="black", rasterized=True)
        ax3.set_xlim(ax.get_xlim())
        ax3.set_ylim(ax.get_ylim())
        ax3.set_xlabel("Time $\\times$ drift velocity (mm)")
        ax3.set_ylabel("Pseudo-channel")
        plt.tight_layout()
        fig3.savefig(uniquify_filename("unclustered-hits-evt-disp-%s.pdf" % plot_label))
        plt.close(fig3)

    return n_clusters, n_cluster_hits, n_noise

def fit_one_set(files, time_window, max_fits, plot_label):
    tmin,tmax=find_time_limits(files)
    print("Times from {} to {}".format(arrow.get(tmin/50e6).format("YYYY-MM-DD HH:mm:ss UTC"),
                                       arrow.get(tmax/50e6).format("YYYY-MM-DD HH:mm:ss UTC")))
    times=np.arange(tmin, tmax, time_window)
    n_clusters=[]
    n_cluster_hits=[]
    n_noises=[]
    nfits=min(max_fits,len(times)-1) if max_fits is not None else len(times)-1
    for i in range(nfits):
        print("Fit {} of {}...".format(i, nfits))
        n_cluster,n_cluster_h,n_noise=find_clusters(files, times[i], times[i+1], plot=(i<5), plot_label=plot_label)
        n_clusters.append(n_cluster)
        n_noises.append(n_noise)
        n_cluster_hits.append(n_cluster_h)

    taxis=(times[0:nfits]-tmin)/50e6
    
    return n_clusters, n_cluster_hits, n_noises, taxis

if __name__=="__main__":
    rcParams["font.size"]=24
    rcParams["figure.figsize"]=[12.8, 9.6]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--on-files', metavar='FILE', type=str, nargs='+',
                    help='Input filenames for neutron source ON')
    parser.add_argument('--off-files', metavar='FILE', type=str, nargs='+',
                        help='Input filenames for neutron source OFF')
    parser.add_argument("--max-fits", default=None, type=int,
                        help="Maximum number of fits to run")
    parser.add_argument("--label", default=None, type=str,
                        help="Label for output files")
    parser.add_argument("--time-window", default=1000000, type=int,
                        help="Time window for each fit in 50 MHz clock ticks")
    parser.add_argument("--batch", action="store_true",
                        help="Don't display anything on screen (useful if saving many event displays to file")

    args=parser.parse_args()

    label="-{}".format(args.label) if args.label else ""
    
    n_clusters_on,  n_cluster_hits_on,  n_noises_on,  taxis_on  = fit_one_set(args.on_files,  args.time_window, args.max_fits, "on")
    n_clusters_off, n_cluster_hits_off, n_noises_off, taxis_off = fit_one_set(args.off_files, args.time_window, args.max_fits, "off")
    
    fig,ax=plt.subplots()
    ax.plot(taxis_on, n_clusters_on,   "o-", label="n source on")
    ax.plot(taxis_off, n_clusters_off, "v-", label="n source off")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of tracks")
    ax.set_ylim(0,100)
    ax.legend()
    plt.tight_layout()
    fig.savefig("number-of-tracks%s.pdf" % label)
        
    fig,ax=plt.subplots()
    ax.plot(taxis_on,  n_cluster_hits_on,   "o-", label="n source on")
    ax.plot(taxis_off, n_cluster_hits_off,  "v-", label="n source off")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of hits in tracks")
    ax.set_ylim(0,3e4)
    ax.legend()
    plt.tight_layout()
    fig.savefig("number-of-track-hits%s.pdf" % label)
    
    fig,ax=plt.subplots()
    ax.plot(taxis_on,  n_noises_on,   "o-", label="n source on")
    ax.plot(taxis_off, n_noises_off,  "v-", label="n source off")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of untracked hits")
    ax.set_ylim(0,1000)
    ax.legend()
    plt.tight_layout()
    fig.savefig("number-of-untracked-hits%s.pdf" % label)
        
    fig,ax=plt.subplots()
    ax.plot(taxis_on,  np.array(n_noises_on)/np.array(n_cluster_hits_on),   "o-", label="n source on")
    ax.plot(taxis_off, np.array(n_noises_off)/np.array(n_cluster_hits_off), "v-", label="n source off")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Untracked hits/Tracked hits")
    ax.set_ylim(0, 0.08)
    ax.legend()
    plt.tight_layout()
    fig.savefig("untracked-tracked-ratio%s.pdf" % label)

    fig,ax=plt.subplots()
    ax.plot(taxis_on,  np.array(n_cluster_hits_on)/np.array(n_clusters_on),   "o-", label="n source on")
    ax.plot(taxis_off, np.array(n_cluster_hits_off)/np.array(n_clusters_off), "v-", label="n source off")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracked hits per track")
    ax.set_ylim(0, 500)
    ax.legend()
    plt.tight_layout()
    fig.savefig("tracked-hits-per-track%s.pdf" % label)

    if not args.batch:
        plt.show()
