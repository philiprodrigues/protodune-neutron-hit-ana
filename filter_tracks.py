import cluster
import argparse
from sklearn.cluster import DBSCAN
import numpy as np

def find_noise_hits(files, tmin, tmax):
    all_hits=cluster.munged_hits_for_times(files, tmin, tmax)
    # This code copied from
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    db = DBSCAN(eps=20, min_samples=5).fit(all_hits)
    labels = db.labels_
    return all_hits[labels==-1]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--in', dest="input", metavar='FILE', type=str, nargs='+',
                    help='Input filenames')
    parser.add_argument('-o','--out', metavar='OUTFILE', type=str,
                        help='Output filename')
    parser.add_argument("--max-fits", default=None, type=int,
                        help="Maximum number of fits to run")

    args=parser.parse_args()

    tmin,tmax=cluster.find_time_limits(args.input)
    times=np.arange(tmin, tmax, 1000000)
    nfits=min(args.max_fits,len(times)-1) if args.max_fits is not None else len(times)-1
    outarr=None
    for i in range(nfits):
        print("Fit {} of {}".format(i, nfits))
        if outarr is None:
            outarr=find_noise_hits(args.input, times[i], times[i+1])
        else:
            outarr=np.vstack((outarr,find_noise_hits(args.input, times[i], times[i+1])))
    np.save(args.out, outarr)
