from multiprocessing.dummy import Process
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
import numpy as np
import scipy.stats as sps
import pandas as pd
from pathlib import Path
import csv

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/Repos/mocap-evaluation/output/Dual/Mixamo/1"))

def plotError(error_array, error_type, dual=False):
    fig = plt.figure()
    mu, std = sps.norm.fit(error_array)
    data = {'error': error_array}
    data = pd.DataFrame(data=data)
    # Plot the histogram
    sns.histplot(data, bins=25, stat="probability")

    if (dual):        

        gm = GaussianMixture(n_components=2, random_state=0).fit(data)
        mu1, mu2 = gm.means_[0], gm.means_[1]
        sigma1, sigma2 = gm.covariances_[0], gm.covariances_[1]
        
        plt.xlabel(error_type + " speed error")
        plt.ylabel("Proportion")
        title = error_type + " speed error results: mu1 = %.2f,  std1 = %.2f" % (mu, std)
        plt.title(title)

        return fig, mu1, mu2, sigma1, sigma2
    else:
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = sps.norm.pdf(x, mu, std)
        p = p/np.linalg.norm(p)
        plt.plot(x, p, 'k', linewidth=2)
        plt.xlabel(error_type + " speed error")
        plt.ylabel("Proportion")
        title = error_type + " speed error results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)
        return fig, mu, std

def read_csv(radius = "", prefix="", sufix="", parents=0):
    info = {}

    p = FOLDER_FILES

    with open(p + '/' + prefix + '/' + str(radius) + "_" + sufix + ".csv", newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        if (sufix == "info"):            
            for row in read_csv:
                info[row[0]] = float(row[1])

    csvfile.close()
        
    return info         


def read_root(filename):
    file = []
    with open(FOLDER_FILES + '/../' + filename + ".csv", newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')        
        for row in read_csv:
            file.append(row[0])
    csvfile.close()
    return file

def ProcessData(prefix, i, radius):
    loaded = np.load(FOLDER_FILES + '/' + prefix + "_processing_dump.npz", allow_pickle=True)
    speed_arrays = loaded['speed_arrays']

    trial_speed = speed_arrays[i + 1]
    
    info = read_csv(abs(radius), prefix=prefix, sufix = "info")   
    linear_error = trial_speed - info["Desired linear speed"]
    lserror_fig, mu1, mu2, sigma1, sigma2 = plotError(linear_error, "Linear", True)
    
    lserror_fig.savefig("C:/Users/vabicheq/Documents/Repos/mocap-evaluation/temp/" + str(radius) + "_linear_speed_error.png")

speed_arrays = []
radiuses = [float(x) for x in read_root("radiuses_short")]

r = iter(radiuses)
c = next(r)

###
i = 0
prefix = "Right"
while c < 0:
    #right 
    ProcessData(prefix, i, c)
    c = next(r)

###
i = 0
prefix = "Straight"
while c == 0.0:
    #straight 
    c = 0
    ProcessData(prefix, i, c)
    c = next(r)

###
i = 0
prefix = "Left"
while c > 0:
    #left 
    ProcessData(prefix, i, c)
    try:
        c = next(r)
    except:
        break