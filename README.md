# Faults Handler

Open-source application of faults detection in 2D seismic images using YOLO11 segmentation model.

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

To train modify configuration in `config/seismic_data.yaml` and run:
```
python src/train.py
```
