# Face Builder

Face Builder is an interactive 3D face modeling application that allows you to manipulate FLAME face models using a simple and intuitive interface. This tool enables you to align a 3D face mesh to images, add control pins, and modify the face shape in various ways.

![Face Builder Interface](screenshots/interface.png)

## Features

- **Interactive 3D Face Manipulation**: Drag landmarks to reshape the 3D face mesh
- **Multi-view Support**: Work with multiple images of the same face from different angles
- **Custom Control Pins**: Add custom pins to control specific areas of the face
- **Automatic Face Alignment**: Automatically align the 3D mesh to detected face landmarks
- **Shape Management**: Reset shape while maintaining position, or completely center geometry
- **Pin Manipulation**: Toggle pin movement mode, add pins, and remove pins as needed
- **OBJ Export**: Save your model as an OBJ file for use in other 3D applications

## Installation

### Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- dlib
- FLAME 2023 model data

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face_builder.git
   cd face_builder

Install dependencies:
bashCopypip install numpy opencv-python dlib

Download the required model files:

Place flame2023.pkl in the data/ directory
Place flame_static_embedding.pkl in the data/ directory
Place shape_predictor_68_face_landmarks.dat in the data/ directory


Prepare your images:

Place your face images in the images/ directory
For best results, include at least a frontal view and side view



Usage

Run the application:
bashCopypython main.py

Interface Controls:

Center Geo: Reset the head to default position
Align Face: Automatically detect and align the 3D mesh to the face
Reset Shape: Restore default shape while maintaining position
Remove Pins: Clear all custom pins from the current image
Toggle Pins: Switch to pin movement mode without affecting the mesh
Add Pins: Add custom control pins to the mesh
Save Mesh: Save the current model to file
Next/Prev Image: Navigate between multiple images


Workflow:

Start with a frontal image and click "Align Face"
Adjust the mesh shape by dragging landmarks
Add custom pins for finer control
Switch to other views to check alignment
Save your model when finished



File Structure
Copyface_builder/
├── main.py               # Main application entry point
├── config.py             # Configuration settings
├── file_io.py            # File operations
├── geometry.py           # Geometric operations
├── ui_dimensions.py      # UI sizing calculations
├── ui_rendering.py       # UI drawing functions
├── ui_input.py           # User input handling
├── model_mesh.py         # Mesh manipulation
├── model_landmarks.py    # Landmark operations
├── model_pins.py         # Pin operations
├── data/                 # Model data files
│   ├── flame2023.pkl
│   ├── flame_static_embedding.pkl
│   └── shape_predictor_68_face_landmarks.dat
├── images/               # Input face images 
└── output/               # Output models and OBJ files
Advanced Usage
Working with Multiple Views
For the most accurate face models, it's recommended to use multiple images from different angles:

First align and adjust the frontal view
Switch to a side view using "Next Image"
Click "Align Face" again to align the model to the side view
The model will maintain the shape adjustments from the frontal view while aligning to the new angle

Custom Pins
Custom pins provide precise control over specific areas:

Click "Add Pins" to enter pin placement mode
Click anywhere on the mesh to add a pin
Use "Toggle Pins" to move pins independently of the mesh
Pins in normal mode will move the mesh when dragged

Exporting Models
Face Builder saves models in two formats:

A pickle file (.pkl) containing all model data
An OBJ file for import into other 3D modeling software

Both are saved to the output/ directory.
Troubleshooting

Face detection fails: Ensure your image has good lighting and a clear view of the face
Mesh appears distorted: Try using the "Reset Shape" button and starting over
Program crashes: Verify that all model files are correctly placed in the data/ directory

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

FLAME face model by MPI-IS
dlib facial landmark detection

Copy
You can copy this entire block of text and save it as README.md in your project's root directory.