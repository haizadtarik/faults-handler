import os
import numpy as np
import cigvis
from cigvis import colormap
import shutil

prediction_dir = "data/prediction/"
seismic_path = prediction_dir + "xs.dat"
fault_path = prediction_dir + "fvt.dat"
skin_dir = prediction_dir + "skins/"

# create skin directory if it doesn't exist and move skins
os.makedirs(skin_dir, exist_ok=True)
for file in os.listdir(prediction_dir):
    if file.startswith("skin") and file.endswith(".dat"):
        shutil.move(os.path.join(prediction_dir, file),
                    os.path.join(skin_dir, file))

ni, nx, nt = 128, 128, 128
shape = (ni, nx, nt)
sx = np.fromfile(seismic_path, dtype=np.float32).reshape(shape)
fx = np.fromfile(fault_path, dtype=np.float32).reshape(shape)
# mask min value (0), 0 means no fault
fg_cmap = colormap.set_alpha_except_min("jet", alpha=1)
# fx is discrete data, set 'interpolation' to 'nearest'
nodes = cigvis.create_slices(sx, pos=[[36], [28], [84]], cmap="gray")
nodes = cigvis.add_mask(nodes, fx, cmaps=fg_cmap, interpolation="nearest")
nodes += cigvis.create_colorbar_from_nodes(nodes, "Amplitude", select="slices")
# fault skin
nodes += cigvis.create_fault_skin(skin_dir, endian=">", values_type="likelihood")
cigvis.plot3D(nodes, size=(700, 600))