import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import csv
import sys
from pathlib import Path
import seaborn as sns
import scipy.stats as sps
from scipy.signal import argrelextrema
import pandas as pd
import os
import itertools
import random
import gc
    
import warnings

# warnings.simplefilter("error", np.VisibleDeprecationWarning)

### ARGUMENTS ###
# <string> CSV file name
# <int> sampling rate

marker = itertools.cycle((',', '+', '.', 'o', '*')) 
extension = "png"

def floatToString(inputValue):
    return ('%.15f' % inputValue).rstrip('0').rstrip('.')

def plotError(error_array, error_type):
    fig = plt.figure()
    mu, std = sps.norm.fit(error_array)
    data = {'error': error_array}
    data = pd.DataFrame(data=data)
    # Plot the histogram
    sns.histplot(data, bins=25, stat="probability")

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

def pause():
    programPause = input("Press the <ENTER> key to continue...")
   
def magnitude(vx, vy): 
    return math.sqrt((vx * vx) + (vy * vy))

def read_csv(radius = "", prefix="", sufix="", parents=0):
    info = {}
    frames = []
    clips = []
    total_frames = []
    switching_statuses = []
    x, y, ry, t = [], [], [], []
    all_x, all_y, all_ry, all_dt = [], [], [], []

    if parents > 0:
        p = str(Path(FOLDER_FILES).parents[parents])
    else:
        p = FOLDER_FILES

    with open(p + '/' + prefix + str(radius) + "_" + sufix + ".csv", newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        if (sufix == "info"):            
            for row in read_csv:
                info[row[0]] = float(row[1])
        if (sufix == "frames"):
            for row in read_csv:
                frames.append(int(row[0]))
        if (sufix == "clips"):
            for row in read_csv:
                clips.append(row[0])
                total_frames.append(int(row[1]))
        if (sufix == "final" or sufix == "dataset"):
            time = 0
            last_size = 0
            for current_size, row in enumerate(read_csv):
                x.append(float(row[1]))
                y.append(float(row[3]))
                ry.append(float(row[5]))
                t.append(float(row[0]))
                if (float(row[0]) < time):
                    all_x.append(np.asarray(x[last_size:current_size]))
                    all_y.append(np.asarray(y[last_size:current_size]))
                    all_ry.append(np.asarray(ry[last_size:current_size]))
                    all_dt.append(np.diff(t[last_size:current_size]))
                    last_size = current_size
                time = float(row[0])

            all_x.append(np.asarray(x[last_size:current_size]))
            all_y.append(np.asarray(y[last_size:current_size]))
            all_ry.append(np.asarray(ry[last_size:current_size]))
            all_dt.append(np.diff(t[last_size:current_size]))
        if (sufix == "switching_status"):
            for row in read_csv:
                switching_statuses.append(row[0])

    csvfile.close()
        
    if (sufix == "info"):   
        return info         
    if (sufix == "frames"):
        return frames
    if (sufix == "clips"):
        return clips, total_frames
    if (sufix == "final" or sufix == "dataset"):
        return np.asarray(all_x, dtype=object), np.asarray(all_y, dtype=object), np.asarray(all_ry, dtype=object), np.asarray(all_dt, dtype=object)
    if (sufix == "switching_status"):
        return switching_statuses

def randomColor():
    return (random.random(), random.random(), random.random())

print(25*'*')
print(sys.argv[0], "start")
print(25*'*')

animation_dataset_file = sys.argv[1]
radiuses = [floatToString(float(x)) for x in sys.argv[2].split(',')]
time_windows = [float(x) for x in sys.argv[3].split(',')]
PLOT_ENABLED = eval(sys.argv[4])

if (len(sys.argv) > 5):
    FOLDER_FILES = str(Path(sys.argv[5]))
    OVERWRITE = eval(sys.argv[6])
    parents = 1
else:
    FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/MotionMatching/Assets/output/Mixamo/1.5"))
    parents = 0

PLOT_AVAILABLE_TRANSITIONS = False

csv_file_name = []
csv_file_name.append(animation_dataset_file)
for r in radiuses:
    csv_file_name.append(str(r) + "_final.csv")

print('*' * 25)
print("Program setup:")
print("Animation dataset file: ", animation_dataset_file)
print("CSV files names: ", csv_file_name[1:])
print("Radiuses: ", radiuses)
print("Time windows: ", time_windows)
print("Plot?: ", PLOT_ENABLED)
print('*' * 25)

radiuses.insert(0, -1) # for the dataset

root_path = str(Path(FOLDER_FILES).parents[1])
speed_arrays = []
theta_arrays = []
clip_arrays = []
dataset_speeds = []
dataset_thetas = []
dt_arrays = []
dataset_raw_ls = []   
dataset_raw_as = []

if (OVERWRITE):
    try:
        os.remove(root_path + "/animation_dataset_dump.npz")
    except:        
        print("animation_dataset_dump.npz already removed!")

for idx, r in enumerate(radiuses):      
    if(os.path.isfile(root_path + "/animation_dataset_dump.npz") and r == -1):
        print("Animation dataset dump loaded.")
    else:
        if (r == -1):
            all_x_arrays, all_y_arrays, all_ry_arrays, all_dt_arrays = read_csv(prefix = "animation", sufix = "dataset", parents=parents)
            clip_indexes = None
            indexesAvailable = False
        else:
            all_x_arrays, all_y_arrays, all_ry_arrays, all_dt_arrays = read_csv(r, sufix = "final")
            loaded = np.load(FOLDER_FILES + '/' + str(r) + "_dominant_clip_every_frame.npz", allow_pickle=True)
            all_clips = loaded['clips']
            clip_names, clip_total_frames = read_csv(prefix = "animation", sufix = "clips", parents=parents)
            clip_indexes = [int(item['AnimId']) for item in all_clips]
            indexesAvailable = True

        # print("r: ", r)
        # print("t size: ", all_dt_arrays.shape)
        # print("X size: ", all_x_arrays.shape)
        # print("Y size: ", all_y_arrays.shape)
        # print("RY size: ", all_ry_arrays.shape, '\n')

        speed = []
        theta = []
        clips = []
        raw_speed = []        
        last_size = 0

        for i in range(0, len(all_dt_arrays)):    
            dx = np.diff(all_x_arrays[i])
            dy = np.diff(all_y_arrays[i])

            if (r == -1):
                for j in range(0, i):
                    if (len(all_x_arrays[i]) == len(all_x_arrays[j])):
                        diff = sum(all_x_arrays[i] - all_x_arrays[j])
                        if (diff == 0):
                            print("Animations", i, "and", j, "are mirrored.")
                            all_x_arrays[i] = -all_x_arrays[i]
                            all_ry_arrays[i] = -all_ry_arrays[i]

            x = [j for j in range(0, len(all_ry_arrays[i]))]

            speed_acc = []
            theta_acc = []
            time_acc = []
            clips_acc = []
            dataset_acc_ls = []
            dataset_acc_as = []
            frames_last = False
            dtheta = np.diff(all_ry_arrays[i])

            dtheta[dtheta >= 180] -= 360
            dtheta[dtheta <= -180] += 360
            
            for j in range(1, len(all_dt_arrays[i])):
                dt = all_dt_arrays[i][j - 1]                  

                if (abs(dtheta[j] / dt) > 510):      
                    print(39*"*")
                    print("Potential problem detected. (Overspeed)")     
                    print("Speed captured:", dtheta[j] / dt)           
                    print("Radius:", r)
                    print("Animation:", i + 1)
                    print("Frame:", j)
                    print(39*"*") 

                if (r == -1):                    
                    dataset_acc_ls.append(magnitude(dx[j - 1] / dt, dy[j - 1] / dt))
                    dataset_acc_as.append(dtheta[j] / dt) 
                else:                    
                    raw_speed.append(magnitude(dx[j - 1] / dt, dy[j - 1] / dt))   

                if (time_windows[idx] > 0):           
                    if (sum(time_acc) <= time_windows[idx]):
                        time_acc.append(dt)
                        speed_acc.append(magnitude(dx[j - 1] / dt, dy[j - 1] / dt))
                        theta_acc.append(dtheta[j] / dt)
                        if (indexesAvailable):
                            clips_acc.append(clip_indexes[j])
                        frames_last = True
                    else:
                        speed.append(np.mean(speed_acc))
                        theta.append(np.mean(theta_acc))
                        if (indexesAvailable):
                            clips.append(clips_acc)
                        frames_last = False
                        while (sum(time_acc) > time_windows[idx]):
                            time_acc.pop(0)
                            speed_acc.pop(0)
                            theta_acc.pop(0) 
                            if (indexesAvailable):
                                clips_acc.pop(0)
                else:
                    speed.append(magnitude(dx[j - 1] / dt, dy[j - 1] / dt))
                    theta.append(dtheta[j] / dt)
                    if (indexesAvailable):
                        clips.append(clip_indexes[j])
                    
            if (frames_last):     
                speed.append(np.mean(speed_acc))
                theta.append(np.mean(theta_acc))
                if (indexesAvailable):
                    clips.append(clips_acc)

            if (r == -1):
                current_size = len(speed)
                dataset_speeds.append(speed[last_size:current_size])
                dataset_thetas.append(theta[last_size:current_size])
                dataset_raw_ls.append(dataset_acc_ls)
                dataset_raw_as.append(dataset_acc_as)
                last_size = current_size
                clips = None

        np.savez_compressed(FOLDER_FILES + '/' + str(r) + "_raw_speed_dump", raw_speed=raw_speed)

        speed_arrays.append(np.asarray(speed))
        theta_arrays.append(np.asarray(theta))
        clip_arrays.append(np.asarray(clips))

        print("Radius", r, "processed.")

if(not os.path.isfile(root_path + "/animation_dataset_dump.npz")):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
        np.savez_compressed(root_path + "/animation_dataset_dump", dataset_speed=speed_arrays[0], dataset_theta=theta_arrays[0])
        np.savez_compressed(root_path + "/raw_dataset", dataset_raw_ls=dataset_raw_ls, dataset_raw_as=dataset_raw_as)
        np.savez_compressed(root_path + "/different_motion_dump", dataset_speeds=dataset_speeds, dataset_thetas=dataset_thetas)
        np.savez_compressed(root_path + "/clips_arrays_dump", clip_arrays=clip_arrays)
else:
    loaded = np.load(root_path + "/animation_dataset_dump.npz", allow_pickle=True)
    speed_arrays.insert(0, loaded['dataset_speed'])
    theta_arrays.insert(0, loaded['dataset_theta'])

    if (not os.path.isfile(root_path + "/different_motion_dump.npz")):
        print("different_motion_dump.npz is missing. Terminating program...")
        exit()
    loaded = np.load(root_path + "/different_motion_dump.npz", allow_pickle=True)
    dataset_speeds = loaded['dataset_speeds']
    dataset_thetas = loaded['dataset_thetas']
    
    if (not os.path.isfile(root_path + "/clips_arrays_dump.npz")):
        print("clips_arrays_dump.npz is missing. Terminating program...")
        exit()        
    loaded = np.load(root_path + "/clips_arrays_dump.npz", allow_pickle=True)
    clip_arrays = loaded['clip_arrays']
    
    if (not os.path.isfile(root_path + "/raw_dataset.npz")):
        print("raw_dataset.npz is missing. Terminating program...")
        exit()        
    loaded = np.load(root_path + "/raw_dataset.npz", allow_pickle=True)
    dataset_raw_ls = loaded['dataset_raw_ls']
    dataset_raw_as = loaded['dataset_raw_as']

file_nbr = len(csv_file_name)

radiuses.remove(-1)

# different motions
different_motions = plt.figure(figsize=(16, 9))
for speeds, thetas in zip(dataset_speeds, dataset_thetas):
    plt.scatter(thetas, speeds, color=randomColor(), marker=next(marker))
    plt.plot(thetas, speeds, color=randomColor())
plt.close(different_motions)

for i, r in enumerate(radiuses):      
    if (os.path.isdir(FOLDER_FILES + '/stack/' + str(r) + '/')):
        pass
    else:
        os.makedirs(FOLDER_FILES + '/stack/' + str(r) + '/')

    loaded = np.load(FOLDER_FILES + '/' + str(r) + "_all_channels.npz", allow_pickle=True)
    all_channels = loaded['all_channels']    
    
    if (PLOT_AVAILABLE_TRANSITIONS):    
        transition_frames = []
        transition_indexes = []
        dominant_frames = []
        dominant_idx = []
        for channels in all_channels:
            frames = []
            indexes = []
            for channel in channels:
                frames.append(round(float(channel['Time']) * 30))
                indexes.append(int(channel['AnimId']))
                if (channel['BlendStatus'] == "Dominant"):
                    dominant_idx.append(int(channel['AnimId']))
                    dominant_frames.append(round(float(channel['Time']) * 30))
            transition_frames.append(frames)
            transition_indexes.append(indexes)        
        
        # available transitions at every frame
        count = 0    
        style = "Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=style, color="red")
        available_transitions = plt.figure(num=1, clear=True, figsize=(16, 16))

        for channel_frames, channel_indexes, dom_frame, dom_idx in zip(transition_frames, transition_indexes, dominant_frames, dominant_idx):
            ax = available_transitions.add_subplot()

            if (dom_frame >= len(dataset_raw_as[dom_idx])):
                dom_frame = len(dataset_raw_as[dom_idx]) - 1
            x_ini, y_ini = dataset_raw_as[dom_idx][dom_frame], dataset_raw_ls[dom_idx][dom_frame]

            plotted_indexes = []
            for frame, index in zip(channel_frames, channel_indexes):   
                if (frame >= len(dataset_raw_as[index])):
                    frame = len(dataset_raw_as[index]) - 1    
                #if (index not in plotted_indexes):
                ax.plot(dataset_raw_as[index], dataset_raw_ls[index], color=randomColor(), zorder=1, alpha=0.5)
                ax.scatter(dataset_raw_as[index], dataset_raw_ls[index], color=randomColor(), marker=next(marker), label=index, zorder=2, alpha=0.5)
                ax.scatter(dataset_raw_as[index][frame], dataset_raw_ls[index][frame], color='lime', zorder=3)            
                x_dst, y_dst = dataset_raw_as[index][frame], dataset_raw_ls[index][frame]
                curved_arrow = patches.FancyArrowPatch((x_ini, y_ini), (x_dst, y_dst), **kw, zorder=3)
                ax.add_patch(curved_arrow)
                plotted_indexes.append(index)

            ax.legend()
            available_transitions.savefig((FOLDER_FILES + '/stack/' + str(r) + '/' + str(count) + '.' + extension))
            available_transitions.clf()
            count += 1

        plt.close(available_transitions)    
        print("Finished saving available transitions!")
    
    # frame and clip occurences in each frame
    loaded = np.load(FOLDER_FILES + '/' + str(r) + "_dominant_clip_every_frame.npz", allow_pickle=True)
    all_clips = loaded['clips']

    clip_indexes = [int(item['AnimId']) for item in all_clips]
    clip_time = np.asarray([float(item['Time']) for item in all_clips])
    clip_frames = np.around(clip_time * 30, decimals=0).astype(int)
    
    # trajectory between frames
    frames_and_clips = {}
    for AnimId, cf in zip(clip_indexes, clip_frames):
        if AnimId in frames_and_clips.keys():
            frames_and_clips[AnimId].append(cf)
        else:
            frames_and_clips[AnimId] = [cf]

    trajectory_between_frames = plt.figure(num=1, clear=True, figsize=(50, 18))
    ax = trajectory_between_frames.add_subplot()
    # ax.set_xlim = [-1, len(max(dataset_raw_as, key=len))]
    # ax.set_ylim = [-1, len(clip_indexes) + 1]
    plot_order = pd.unique(clip_indexes)
    ticks = []
    labels = []

    max_color = -9999999999999999
    min_color = +9999999999999999
    for j, clip in enumerate(plot_order):
        number_of_frames_in_clip = len(dataset_raw_as[clip])

        # plot whole animation
        y_plot = [j] * number_of_frames_in_clip

        colors = []

        for k in range(0, number_of_frames_in_clip):
            frames = frames_and_clips[clip]
            if k in frames:
                freq = frames.count(k)
                colors.append(freq)
            else:
                colors.append(0)

        ax.scatter(range(number_of_frames_in_clip), y_plot, c=colors, cmap='plasma', s=2, marker='.', zorder=3)

        ticks.append(j)
        labels.append(clip)

        if (max_color < max(colors)):
            max_color = max(colors)
        if (min_color > min(colors)):
            min_color = min(colors)

    norm = mpl.colors.Normalize(vmin=min_color,vmax=max_color)
    trajectory_between_frames.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=ax)

    ax.set_yticks(ticks, labels)

    style = "Simple, tail_width=0.5, head_width=4, head_length=8"

    x_pair_list = []
    y_pair_list = []

    for j in range(1, len(clip_frames)):
        # plot connections
        kw = dict(arrowstyle=style, color="red")
        x_ini = clip_frames[j - 1]
        y_ini = np.where(plot_order == clip_indexes[j - 1])[0][0]
        x_dst = clip_frames[j]
        y_dst = np.where(plot_order == clip_indexes[j - 1])[0][0]

        if (clip_indexes[j] != clip_indexes[j - 1]):
            kw = dict(arrowstyle=style, color="blue")   
            y_dst = np.where(plot_order == clip_indexes[j])[0][0]

        x_pair = (x_ini, x_dst)
        y_pair = (y_ini, y_dst)

        x_pair_list.append(x_pair)
        y_pair_list.append(y_pair)


        lw = (x_pair_list.count(x_pair) + x_pair_list.count(y_pair)) / 2

        curved_arrow = patches.FancyArrowPatch((x_ini, y_ini), (x_dst, y_dst), connectionstyle="arc3,rad=.5", **kw, lw = lw)
        ax.add_patch(curved_arrow)

    ax.autoscale_view()
    trajectory_between_frames.set_tight_layout(True)
    trajectory_between_frames.savefig((FOLDER_FILES + "/images/"  + str(r) + "_trajectory_between_frames." + extension))

gc.collect()

for i, r in enumerate(radiuses):
    info = read_csv(r, sufix = "info")
    frames = read_csv(r, sufix = "frames")
    switching_statuses = read_csv(r, sufix = "switching_status")
    switching_statuses = np.asarray(switching_statuses, dtype=np.int32)

    switching_speeds = switching_statuses * info["Switching linear speed"]
    switching_speeds[switching_speeds == 0] = info["Desired linear speed"]

    if (not os.path.isfile(FOLDER_FILES + '/' + str(r) + "_raw_speed_dump.npz")):
        print(str(r) + "raw_speed_dump.npz is missing. Terminating program...")
        exit()        
    loaded = np.load(FOLDER_FILES + '/' + str(r) + "_raw_speed_dump.npz", allow_pickle=True)
    trial_raw_speed = np.asarray(loaded['raw_speed'], dtype=np.float32)

    dataset_speed = speed_arrays[0]
    dataset_theta = theta_arrays[0]
    trial_speed = speed_arrays[i + 1]
    trial_theta = theta_arrays[i + 1]

    # Convolution
    if (r != 0):
        s = np.where(np.diff(switching_speeds))[0] + 1
        offset = round(s[1] / 4)
        simplified_sw_st = switching_speeds[offset:(s[1] + offset)]

        convolution_signals = plt.figure(figsize=(16,9))
        plt.plot(range(0, len(switching_speeds)), switching_speeds, label='Switching')
        plt.plot(range(0, len(trial_raw_speed)), trial_raw_speed, label='Raw speed')
        plt.plot(range(0, len(simplified_sw_st)), np.flip(simplified_sw_st), label='Simplified')
        plt.legend()
        convolution_signals.savefig(FOLDER_FILES + '/images/' + str(r) + "_convolution_signals." + extension)
        plt.close(convolution_signals)
        

        convolution = np.convolve(trial_raw_speed, simplified_sw_st, mode='valid')
        convolution_plot = plt.figure(figsize=(16, 9))
        plt.plot(range(0, len(convolution)), convolution, label='Convolution')
        local_maximas = argrelextrema(convolution, np.greater, order = 100)[0]
        local_minimas = argrelextrema(convolution, np.less, order = 100)[0]

        labels_maximas = []
        labels_minimas = []

        for i, (maxima, minima) in enumerate(zip(local_maximas, local_minimas)):
            labels_maximas.append(abs(s[i] - maxima - offset) / 30) # 30 fps
            labels_minimas.append(abs(s[i + 1] - minima - offset) / 30)

        plt.scatter(local_maximas, convolution[local_maximas], label='Maximas')
        plt.scatter(local_minimas, convolution[local_minimas], label='Minimas')

        ax = plt.gca()

        for txt, x, y in zip(labels_maximas, local_maximas, convolution[local_maximas]):
            ax.annotate(txt, (x, y))
            
        for txt, x, y in zip(labels_minimas, local_minimas, convolution[local_minimas]):
            ax.annotate(txt, (x, y))

        plt.legend()
        convolution_plot.savefig(FOLDER_FILES + '/images/' + str(r) + "_convolution_plot." + extension)
        
        plt.close(convolution_plot)

    # Calculate speed error
    # Angular
    angular_error = trial_theta - info["Angle speed"]
    aerror_fig, mu_angular, std_angular = plotError(angular_error, "Angular")

    # Linear
    linear_error = trial_speed - info["Desired linear speed"]
    lserror_fig, mu_linear, std_linear = plotError(linear_error, "Linear")

    # Plot the dataset
    dataset_plot = plt.figure(figsize=(16, 9)) 
    plt.rc('axes', axisbelow=True)
    plt.grid(color='grey', linestyle='--', linewidth=1)
    data_dict = {'theta': dataset_theta, 'speed': dataset_speed}
    data = pd.DataFrame(data=data_dict)
    xy = np.vstack([dataset_theta, dataset_speed])
    z = sps.gaussian_kde(xy)(xy) 
    coverage = sns.scatterplot(data=data, x='theta', y='speed', c=z, cmap='mako')
    path_coverage = str(Path(FOLDER_FILES).parents[1])
    np.savez_compressed(path_coverage + "/coverage_plot", data=data_dict, z=z)

    # Plot the desired point
    desiredPoint = [info["Angle speed"], info["Desired linear speed"]]
    plt.errorbar(desiredPoint[0], desiredPoint[1], xerr=std_angular, yerr=std_linear, color='red', capsize=5.0)
    coverage.scatter(desiredPoint[0], desiredPoint[1], color='lime', zorder=2)

    np.savez_compressed(FOLDER_FILES + "/" + str(r) + "_desiredPoint_and_error", dp=desiredPoint, error=[std_linear, std_angular])

    speed = np.asarray(trial_speed, dtype=object)
    theta = np.asarray(trial_theta, dtype=object)

    fig_ts = plt.figure()
    plt.ylabel("Turning speed (degrees/s)")
    plt.title("Average turning speed (degrees/s) vs time (s) for " + str(time_windows[i]) + "s")
    plt.plot([j for j in range(0, len(theta))], theta)
    plt.plot([l for l in range(0, len(theta))], [theta.mean() for m in range(0, len(speed))])
    #plt.scatter(frames, theta[frames], color='r')
    plt.legend(["Turning speed", "Average", "Transitions"])

    fig_ls = plt.figure()
    plt.ylabel("Linear speed (m/s)")
    plt.title("Average linear speed (m/s) vs time (s) for " + str(time_windows[i]) + "s")
    plt.plot([k for k in range(0, len(speed))], speed)
    plt.plot([l for l in range(0, len(speed))], [speed.mean() for m in range(0, len(speed))])
    #plt.scatter(frames, speed[frames], color='r'
    plt.legend(["Linear speed", "Average", "Transitions"])

    if (os.path.isdir(FOLDER_FILES + '/images/')):
        pass
    else:
        os.mkdir(FOLDER_FILES + '/images/')

    coverage.figure.savefig(FOLDER_FILES + '/images/' + str(r) + "_coverage." + extension)
    fig_ts.savefig(FOLDER_FILES + '/images/' + str(r) + "_turning_speed." + extension)
    fig_ls.savefig(FOLDER_FILES + '/images/' + str(r) + "_linear_speed." + extension)
    aerror_fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_angular_speed_error." + extension)
    lserror_fig.savefig(FOLDER_FILES + '/images/' + str(r) + "_linear_speed_error." + extension)
    different_motions.savefig(FOLDER_FILES + '/images/' + str(r) + "_different_motions." + extension)

    trial_speed = np.asarray(trial_speed, dtype=object)
    
    file = open(FOLDER_FILES + '/' + str(r) + "_speed_dump.csv", 'w', newline='')

    writer = csv.writer(file)

    for speed in trial_speed:
        writer.writerow([speed])
    
    file.close()
    plt.close('all')
    gc.collect()

    if (PLOT_ENABLED):
        plt.show()

print(25*'*')
print(sys.argv[0], "end")
print(25*'*')