import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.unet3d.data_loader import SeismicData
from src.unet3d.picker import FaultsPicker  

seismic_data_loader = SeismicData(use_osv=True)
train_seismic_dir = 'data/train/seis'
train_faults_dir = 'data/train/fault'
train_dataset = seismic_data_loader.load_dataset(train_seismic_dir, train_faults_dir)

val_seismic_dir = 'data/validation/seis'
val_faults_dir = 'data/validation/fault'
val_dataset = seismic_data_loader.load_dataset(val_seismic_dir, val_faults_dir)

picker = FaultsPicker(use_osv=True)
picker.train(train_dataset, 
            val_dataset, 
            num_epochs=100,  
            hpo=True,
            )
