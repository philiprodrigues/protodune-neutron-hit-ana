import numpy as np
import argparse
import sys

def maplink(link):
    i=link//5
    j=link%5
    return 5*i+(4-j)

def mapchans(c, minc):
    link=(c-minc)//48
    linkind=(c-minc)%48
    mappedlink=maplink(link)
    return 48*mappedlink+linkind

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--in', dest="input", metavar='FILE', type=str, nargs='+',
                    help='Input filenames')
    parser.add_argument("-r", "--rearrange-channels", action="store_true",
                        help="Remap the channels. Requires --min-ch")
    parser.add_argument("--min-ch", type=int, default=None)
    
    args=parser.parse_args()

    if args.rearrange_channels and not args.min_ch:
        print("--rearrange-channels requires --min-ch")
        sys.exit(1)
        
    for i in args.input:
        print(i)
        a=np.loadtxt(i, dtype=int, usecols=(1,2,3,4))
        if args.rearrange_channels:
            tmp=mapchans(a[:,0], args.min_ch)
            a[:,0]=tmp
        np.save(i.replace(".txt", ".npy"), a)
