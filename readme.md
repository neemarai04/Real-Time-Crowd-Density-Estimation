# Real-Time Crowd Density Estimation using CSRNet

## 1. Project Abstract

This application provides a robust solution for real-time crowd counting and density analysis from image and video sources. It leverages a deep learning model based on the **CSRNet (Congested Scene Recognition Network)** architecture to generate high-fidelity density maps, enabling accurate estimations of crowd size. The system features a configurable zonal analysis module that monitors specific areas of interest and issues alerts based on predefined density thresholds, making it an effective tool for crowd management and safety monitoring.

The front-end is delivered through an interactive web interface built with Streamlit, ensuring ease of use and clear visualization of results.

***

## 2. Key Features

-   **Multi-Source Input:** Supports analysis of both static images (`JPG`, `PNG`) and video files (`MP4`, `AVI`, `MOV`).
-   **Advanced Deep Learning Model:** Employs a **CSRNet** model implemented in Keras to generate precise density maps, ideal for handling highly congested scenes.
-   **Zonal Density Monitoring:** Allows for the definition of multiple, distinct zones within the frame for granular analysis.
-   **Threshold-Based Alerting System:** Triggers multi-level alerts (Safe, Caution, Alert) for each zone when crowd counts exceed user-defined thresholds.
-   **Interactive Visualization:** Renders a comprehensive output including the source video with annotated zones, real-time crowd counts, and a corresponding heat map for intuitive density assessment.

***

## 3. Technical Architecture

The system's processing pipeline follows a sequential workflow:

1.  **Frame Ingestion:** An image or video frame is captured from the user-uploaded source.
2.  **Preprocessing:** The frame is resized, normalized, and standardized to meet the input requirements of the neural network.
3.  **Model Inference:** The preprocessed frame is passed through the pre-trained **CSRNet model**, which outputs a crowd density map.
4.  **Post-processing & Analysis:** The total crowd count is estimated by summing the values in the density map. The map is then segmented according to the defined zones, and zone-specific counts are calculated.
5.  **Status Evaluation:** Each zone's count is compared against its configured thresholds to determine the current status (Safe, Caution, or Alert).
6.  **Output Visualization:** The results are rendered on the Streamlit front-end, displaying the annotated video frame and the generated density map.

***

## 4. Technology Stack

-   **Backend Framework:** Python 3.8+
-   **Web Interface:** Streamlit
-   **Machine Learning:** Keras (TensorFlow backend)
-   **Image/Video Processing:** OpenCV, Pillow
-   **Numerical Computation:** NumPy

***

## 5. Installation and Execution

### 5.1. Prerequisites

-   Python 3.8 or higher
-   Git
-   Git Large File Storage (LFS)

Ensure Git LFS is installed and initialized to handle the large model weights file.
```bash
# Install Git LFS on your system (e.g., using 'brew', 'apt', or from the official website)
git lfs install
```

### 5.2. Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Ayushpk01/PBL.git](https://github.com/Ayushpk01/PBL.git)
    cd PBL
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following contents:
    ```txt
    streamlit
    opencv-python-headless
    numpy
    Pillow
    tensorflow
    matplotlib
    ```
    Install the packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify Project Structure:**
    Confirm that the model architecture and weights are located in the correct directories:
    ```
    /
    ├── models/
    │   └── Model.json
    ├── weights/
    │   └── model_A_weights.h5
    └── app.py
    ```

### 5.3. Running the Application

Execute the following command in the root directory of the project:
```bash
streamlit run app.py
```
The application will be accessible via a local URL displayed in the terminal.

***

## 6. System Configuration

The primary configuration for zonal analysis is managed through the `ZONES` list within the main script. Users can modify this structure to define custom monitoring areas and alerting criteria.

**Example Configuration:**
```python
ZONES = [
    {"name": "Zone 1", "rect": (0, 0, 320, 480), "thresholds": {"Caution": 10, "Alert": 20}},
    {"name": "Zone 2", "rect": (320, 0, 320, 480), "thresholds": {"Caution": 15, "Alert": 25}},
]
```
-   `name`: A unique identifier for the zone.
-   `rect`: A tuple `(x, y, width, height)` defining the zone's bounding box.
-   `thresholds`: A dictionary specifying the crowd count that triggers `Caution` and `Alert` statuses.

***

## 7. Team Contributions

This project was developed through a collaborative effort, with each team member responsible for specific components of the pipeline:

-   **[Ayush](https://github.com/Ayushpk01):**
    -   **Frame Ingestion:** Implemented the file handling logic for user-uploaded images and videos.
    -   **Preprocessing:** Developed the functions to resize, normalize, and standardize frames for model compatibility.
    -   **Component Integration:** Ensured seamless data flow between all parts of the application.

-   **Nireeksha:**
    -   **Model Inference:** Integrated the pre-trained CSRNet model and managed the prediction pipeline.

-   **Neema:**
    -   **Post-processing & Analysis:** Engineered the logic to calculate total and zone-specific crowd counts from the density map.
    -   **Status Evaluation:** Implemented the threshold-based system to determine the status of each monitored zone.

-   **Dhanush:**
    -   **Output Visualization:** Designed and developed the Streamlit user interface, including the rendering of annotated frames and density maps.

***

## 8. License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
