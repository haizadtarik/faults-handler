# Faults Handler

Faults—fractures within the Earth's crust—play a crucial role in the migration and trapping of hydrocarbons. They can either facilitate the movement of hydrocarbons into reservoirs or act as barriers that impede their flow. Accurate identification and interpretation of these faults are essential for geologists involved in oil and gas exploration. In three-dimensional (3D) seismic data, faults are relatively easier to interpret due to the continuous spatial coverage across the area of interest. However, in frontier exploration areas, geologists often have access only to two-dimensional (2D) seismic data, which consists of discrete inline and crossline sections. This lack of continuity makes it challenging to identify and correlate the same fault structures across different 2D lines, especially in regions with complex faulting.

This project aims to address these challenges by developing open-source web application for effective fault detection in 2D seismic images using AI model. Currently, the application run using YOLOv11 segmentation model

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

## Train on Your Own Data

Data need to be in yolo format. For more details, refer [here](https://docs.ultralytics.com/datasets/segment/)

To generate labels file from mask images, use function `create_yolo_segmentation_labels_from_directory` in `util.py`

To train modify configuration in `config/seismic_data.yaml` and `train.py` and run:
```
python src/train.py
```
