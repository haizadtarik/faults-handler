# Faults Handler

Open-source application of faults detection in 2D seismic images using yoloV11 segmentation model.

## Setup

1. Clone this repository
    """
    git clone https://github.com/haizadtarik/faults-handler.git
    cd faults-handler
    """

2. Install necessary dependencies
    ```
    pip install -r requirements.txt
    ```

## Run application

1. Launch server
    ```
    python src/server.py
    ```

2. Launch application

## Train on Your Own Data

Data need to be in yolo format. For more details, refer [here](https://docs.ultralytics.com/datasets/segment/)

To generate labels file from mask images, use function `create_yolo_segmentation_labels_from_directory` in `util.py`

To train modify configuration in `config/seismic_data.yaml` and run:
```
python src/train.py
```