# Faults Handler

Faults—fractures within the Earth's crust—play a crucial role in the migration and trapping of hydrocarbons. They can either facilitate the movement of hydrocarbons into reservoirs or act as barriers that impede their flow. Accurate identification and interpretation of these faults are essential for geologists involved in oil and gas exploration. In three-dimensional (3D) seismic data, faults are relatively easier to interpret due to the continuous spatial coverage across the area of interest. However, in frontier exploration areas, geologists often have access only to two-dimensional (2D) seismic data, which consists of discrete inline and crossline sections. This lack of continuity makes it challenging to identify and correlate the same fault structures across different 2D lines, especially in regions with complex faulting. This project aims to address these challenges by developing open-source web application for effective fault detection in 2D seismic images using AI model. Currently, the application run using YOLOv11 segmentation model

This project now is extended 


## Setup

1. Clone this repository
    ```
    git clone https://github.com/haizadtarik/faults-handler.git
    cd faults-handler
    ```

2. Install necessary dependencies
    ```
    pip install -r requirements.txt
    cd client
    npm install --force
    ```

## Run application

### 2D fault detection using yolo

1. Launch server
    ```
    python src/server.py
    ```

2. On new terminal, launch client application
    ```
    cd client
    npm run dev
    ```

3. Go to [http://localhost:3000](http://localhost:3000) and upload your 2D seismic images to identify the faults

#### Train yolo on Your Own Data

Data need to be in yolo format. For more details, refer [here](https://docs.ultralytics.com/datasets/segment/)

To generate labels file from mask images, use function `create_yolo_segmentation_labels_from_directory` in `src/yolo2d/util.py`

To train modify configuration in `config/seismic_data.yaml` and `src/yolo2d/train.py` and run:
```
python src/train.py
```

### 3D fault detection using U-Net and OSV

1. Setup OSV docker container 
```
git clone https://github.com/haizadtarik/osv.git
cd osv
git checkout osv_unet
docker build -t osv .
```

2. Install cigvis from fork
```
git clone https://github.com/haizadtarik/cigvis.git
pip install -e ".[all]" --config-settings editable_mode=compat
```

2. Train unet model

    a. Download synthetic data from [here](https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8)
    b. Create data directory and arrange the downloaded data like below:
        ```
        data
        |--train
        |--fault
            |--0.dat
            |-- ...
            |--199.dat
        |--seis
            |--0.dat
            |-- ...
            |--199.dat
        |--validation
        |--fault
            |--0.dat
            |-- ...
            |--19.dat
        |--seis
            |--0.dat
            |-- ...
            |--19.dat
        ```
    b. Run the following command:
    ```
    python src/unet3d/train.py
    ```

3. Go through the notebook [here](notebook/unet_osv.ipynb) to run inference using U-Net + OSV

## References
1. [Faultseg](https://github.com/xinwucwp/faultSeg)
2. [OSV](https://github.com/xinwucwp/osv)
