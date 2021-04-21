from pynimation.io import load
from pynimation.viewer import Viewer
from pynimation.common import data as data_
from pynimation.io.fbx import FBXExporter
import pynimation.common.graph as g;
import csv

mvnx_file = data_.getDataPath("data/animations/mocap.mvnx")

[animation] = load(mvnx_file)

csv_file = open('root_mocap.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(["frame", "root x", "root y", "root z"])

for i, frame in enumerate(animation.frames):
    root = frame.getBoneGlobalPosition(frame.skeleton.root.id)
    writer.writerow([i, root[0], root[1], root[2]])

csv_file.close()