
import streamlit as st
# At the top of your app after imports
st.set_page_config(
    page_title="Smart Traffic Management",
    layout="wide",
    initial_sidebar_state="expanded"
)
import numpy as np
import os
import torch
from inference import get_model
import supervision as sv
# try:
#     import torch
#     PYTORCH_AVAILABLE = True
#     st.sidebar.success("✅ PyTorch loaded")
# except Exception as e:
#     st.sidebar.warning(f"⚠️ PyTorch not available: {e}")
#     class DummyTorch:
#         def __init__(self):
#             self.cuda = type('cuda', (), {'is_available': lambda: False})()
#     torch = DummyTorch()

# try:
#     from inference import get_model
# except Exception as e:
#     st.sidebar.warning(f"⚠️ Inference package not available: {e}")
#     get_model = None

# try:
#     import supervision as sv
# except Exception as e:
#     st.sidebar.warning(f"⚠️ Supervision package not available: {e}")
#     sv = None

# Try to import OpenCV
try:
    import cv2
except ImportError as e:
    st.error(f"❌ Failed to import OpenCV: {e}. Please make sure 'opencv-python-headless' is in requirements.txt.")

# # Try to import numpy
# import numpy as np

# # # Handle PyTorch with fallback
# # PYTORCH_AVAILABLE = False
# # try:
# #     import torch
# #     PYTORCH_AVAILABLE = True
# #     st.sidebar.success("✅ PyTorch loaded")
# # except ImportError as e:
# #     st.sidebar.warning(f"⚠️ PyTorch not available: {e}. Some features may be limited.")
# #     # Create dummy torch module to prevent errors
# #     class DummyTorch:
# #         def __init__(self):
# #             self.cuda = type('cuda', (), {'is_available': lambda: False})()
# #     torch = DummyTorch()

# # Try importing other packages
# # try:
# #     from inference import get_model
# # except ImportError as e:
# #     st.sidebar.warning(f"⚠️ Inference package not available: {e}")
# #     get_model = None

# # try:
# #     import supervision as sv
# # except ImportError as e:
# #     st.sidebar.warning(f"⚠️ Supervision package not available: {e}")
# #     sv = None

# import os




# Custom CSS for modern UI
st.markdown("""
<style>
    /* MODERN COLOR PALETTE */
    :root {
        --primary: #4361EE;
        --primary-light: #4895EF;
        --secondary: #3F37C9;
        --accent: #F72585;
        --background: #F8F9FA;
        --sidebar: #FFFFFF;
        --card: #FFFFFF;
        --text: #1E293B;
        --text-light: #64748B;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
    }

    /* GLOBAL STYLES */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-weight: 700;
        color: var(--text);
    }

    /* CUSTOM HEADER */
    .dashboard-header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .header-title {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
    }

    .header-subtitle {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.9;
    }

    /* CARDS */
    .card {
        background: var(--card);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #E2E8F0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }

    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }

    .card-icon {
        background: var(--primary-light);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-size: 1.2rem;
    }

    .card-title {
        margin: 0;
        font-size: 1.2rem;
    }

    /* STATUS INDICATORS */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-normal {
        background: rgba(16, 185, 129, 0.15);
        color: var(--success);
    }

    .status-caution {
        background: rgba(245, 158, 11, 0.15);
        color: var(--warning);
    }

    .status-alert {
        background: rgba(239, 68, 68, 0.15);
        color: var(--danger);
    }

    /* METRIC CARDS */
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: var(--card);
        flex: 1;
        min-width: 200px;
        border-radius: 10px;
        padding: 1.25rem;
        border-left: 4px solid var(--primary);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }

    .metric-title {
        font-size: 0.9rem;
        color: var(--text-light);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }

    .metric-title i {
        margin-right: 5px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-trend {
        display: flex;
        align-items: center;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    .trend-up {
        color: var(--success);
    }

    .trend-down {
        color: var(--danger);
    }

    /* CUSTOM BUTTONS */
    .custom-button {
        display: inline-block;
        background: var(--primary);
        color: white;
        border: none;
        padding: 0.5rem 1.25rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
        margin: 0.25rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .custom-button:hover {
        background: var(--secondary);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    .custom-button.secondary {
        background: white;
        color: var(--primary);
        border: 1px solid var(--primary);
    }

    .custom-button.secondary:hover {
        background: #F8FAFC;
    }

    .custom-button.danger {
        background: var(--danger);
    }

    .custom-button.danger:hover {
        background: #DC2626;
    }

    /* SIDEBAR STYLING */
    .css-1d391kg, .css-163ttbj {
        background-color: var(--sidebar);
    }

    /* TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        border-radius: 10px;
        overflow: hidden;
        background-color: #EEF2FF;
        padding: 0px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0rem 1rem;
        color: var(--text);
        background-color: transparent;
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: var(--primary) !important;
        font-weight: 600;
        border-right: 1px solid #EEF2FF;
        border-left: 1px solid #EEF2FF;
    }

    /* SIMULATION UI */
    .simulation-container {
        background: linear-gradient(to right, #EEF2FF, #F8FAFC);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }

    .traffic-light-container {
        position: relative;
        background: #333;
        width: 120px;
        height: 300px;
        border-radius: 12px;
        margin: 0 auto;
        padding: 15px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    .traffic-light {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: #555;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }

    .traffic-light.red.active {
        background: radial-gradient(circle, #ff0000 40%, #990000 100%);
        box-shadow: 0 0 20px #ff0000;
    }

    .traffic-light.yellow.active {
        background: radial-gradient(circle, #ffff00 40%, #999900 100%);
        box-shadow: 0 0 20px #ffff00;
    }

    .traffic-light.green.active {
        background: radial-gradient(circle, #00ff00 40%, #009900 100%);
        box-shadow: 0 0 20px #00ff00;
    }

    /* LOADER ANIMATION */
    .loader {
        border: 5px solid #f3f3f3;
        border-top: 5px solid var(--primary);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* RESPONSIVE ADJUSTMENTS */
    @media screen and (max-width: 768px) {
        .header-title {
            font-size: 1.8rem;
        }

        .metric-card {
            min-width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)
# Set page configuration
# st.set_page_config(
#     page_title="AI Traffic Management System",
#     page_icon="🚦",
#     layout="wide"
# )

# Define paths for models
HOME = os.getcwd()
VEHICLE_WEIGHTS = os.path.join(HOME, "yolov7.weights")
VEHICLE_CONFIG = os.path.join(HOME, "yolov7.cfg")
COCO_NAMES = os.path.join(HOME, "coco.names")

# App title and description
# Dashboard Header
st.markdown("""
<div class="dashboard-header">
    <div>
        <h1 class="header-title">🚦 Smart Traffic Management</h1>
        <p class="header-subtitle">AI-powered traffic optimization and emergency vehicle prioritization</p>
    </div>
    <div>
        <span class="status-badge status-normal">System Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Function to load YOLO model for vehicle detection
@st.cache_resource
def load_vehicle_model():
    try:
        net = cv2.dnn.readNet(VEHICLE_WEIGHTS, VEHICLE_CONFIG)
        with open(COCO_NAMES, "r") as f:
            classes = f.read().strip().split("\n")
        return net, classes
    except Exception as e:
        st.error(f"Error loading vehicle detection model: {e}")
        return None, None


# Function to load Roboflow model for ambulance detection
@st.cache_resource
def load_ambulance_model(api_key):
    try:
        # Use your actual model ID from Roboflow
        model_id = "ambulance-4bova/1"
        model = get_model(model_id=model_id, api_key=api_key)
        return model
    except Exception as e:
        st.error(f"Error loading ambulance detection model: {e}")
        return None


# Function to detect vehicles with improved motorcycle detection
def detect_vehicles(image, net, classes):
    # Convert image format if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    height, width = image.shape[:2]

    # Improved preprocessing - use a larger size for better detection of small vehicles
    blob = cv2.dnn.blobFromImage(image, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    # Lists for storing detection data
    boxes = []
    class_ids = []
    confidences = []
    labels = []
    vehicle_labels = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']  # Include two-wheelers

    # Track class-specific detections for debugging
    class_counts = {label: 0 for label in vehicle_labels}

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Much lower confidence threshold to catch more vehicles
            if confidence > 0.25:  # Reduced from 0.5
                label = classes[class_id]

                # Handle class confusions with custom mapping
                if label == 'truck' and confidence < 0.8:
                    # Check if it might be a bus
                    bus_id = classes.index('bus') if 'bus' in classes else -1
                    if bus_id >= 0 and scores[bus_id] > 0.2:
                        # It could be a bus, mark as "large vehicle"
                        label = 'large vehicle'

                if label in vehicle_labels or label == 'large vehicle':
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Store all details
                    boxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    labels.append(label)
                    class_counts[label if label in class_counts else 'large vehicle'] += 1

    # Apply non-max suppression with lower IOU threshold to keep more distinct vehicles
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.2)  # More permissive values

    # Draw bounding boxes and count vehicles
    output_image = image.copy()
    vehicles_detected = 0
    detected_labels = []  # To track what was actually drawn

    if len(indices) > 0:
        # Handle different versions of OpenCV
        if isinstance(indices, tuple):
            indices = indices[0]

        # Convert to iterable if needed
        indices_iter = indices.flatten() if hasattr(indices, 'flatten') else indices

        for i in indices_iter:
            x, y, w, h = boxes[i]
            label = labels[i]
            confidence = confidences[i]
            detected_labels.append(label)

            # Apply different colors based on vehicle type
            if label == 'car':
                color = (0, 255, 0)  # Green
            elif label == 'motorcycle' or label == 'bicycle':
                color = (255, 0, 0)  # Blue
            elif label == 'large vehicle':
                color = (255, 165, 0)  # Orange for ambiguous large vehicles
            else:  # bus or truck
                color = (0, 0, 255)  # Red

            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_image, f"{label} {confidence:.2f}",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            vehicles_detected += 1

    # Print detection stats (can be removed in production)
    print(f"Raw detections by class: {class_counts}")
    print(f"After NMS - {vehicles_detected} vehicles: {detected_labels}")

    return output_image, vehicles_detected
# Function to detect ambulances
def detect_ambulances(image, model):
    try:
        # Convert OpenCV image (BGR) to RGB for the model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform inference
        result = model.infer(image_rgb, confidence=0.5)[0]
        detections = sv.Detections.from_inference(result)

        # Create annotators
        label_annotator = sv.LabelAnnotator(text_color=sv.Color.WHITE)
        bounding_box_annotator = sv.BoundingBoxAnnotator(color=sv.Color.RED)

        # Annotate the image
        annotated_image = image_rgb.copy()
        annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # Convert back to BGR for OpenCV display
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Count ambulances
        ambulances_detected = len(detections)

        return annotated_image, ambulances_detected, detections
    except Exception as e:
        st.error(f"Error in ambulance detection: {e}")
        return image, 0, None


# Function to adjust traffic signal timings
def adjust_signal_timings(vehicle_count, ambulance_detected):
    # Default signal timings
    green_time = 30
    red_time = 20
    yellow_time = 5

    # Adjust based on vehicle count
    if vehicle_count > 50:
        green_time = min(green_time + 10, 60)  # Increase green time, max 60s
    elif vehicle_count < 10:
        green_time = max(green_time - 5, 15)  # Decrease green time, min 15s

    # Prioritize ambulance
    if ambulance_detected:
        green_time = 60  # Maximum green time for emergency vehicles
        red_time = 10  # Minimum red time

    return green_time, red_time, yellow_time


# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Roboflow API Key", value="RnqWUZ4KoV9KdlODUpMM", type="password")

    st.markdown("---")
    st.subheader("Traffic Signal Control")

    # Placeholders for debugging
    st.checkbox("Show detection details", key="show_debug")

    # Signal control demonstration
    if st.button("Optimize Signal Timings"):
        st.session_state.optimize = True

# Main application flow
uploaded_file = st.file_uploader("Upload a traffic image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)

    # Display the original image
    st.subheader("Original Image")
    st.image(original_image, channels="BGR", use_column_width=True)

    # Load models
    vehicle_net, vehicle_classes = load_vehicle_model()

    # Create columns for the detection results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vehicle Detection")
        if vehicle_net is not None:
            vehicle_image, vehicle_count = detect_vehicles(original_image.copy(), vehicle_net, vehicle_classes)
            st.image(vehicle_image, channels="BGR", use_column_width=True)
            st.success(f"Detected {vehicle_count} vehicles")

            # Debug information if enabled
            if st.session_state.show_debug:
                st.write("Vehicle Classes in Model:")
                for i, cls in enumerate(vehicle_classes):
                    if cls in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']:
                        st.write(f"{i}: {cls}")
        else:
            st.warning("Vehicle detection model not loaded")

    with col2:
        st.subheader("Ambulance Detection")
        if api_key:
            ambulance_model = load_ambulance_model(api_key)
            if ambulance_model is not None:
                ambulance_image, ambulance_count, detections = detect_ambulances(original_image.copy(), ambulance_model)
                st.image(ambulance_image, channels="BGR", use_column_width=True)
                st.success(f"Detected {ambulance_count} ambulances")
            else:
                st.warning("Ambulance detection model not loaded")
        else:
            st.warning("Please enter your Roboflow API key in the sidebar")

    # Traffic signal optimization
    st.markdown("---")
    st.subheader("Traffic Signal Optimization")

    # Only calculate if both detections were successful
    if 'vehicle_count' in locals() and 'ambulance_count' in locals():
        green_time, red_time, yellow_time = adjust_signal_timings(vehicle_count, ambulance_count > 0)

        # Display signal timings
        cols = st.columns(3)
        with cols[0]:
            st.metric("Green Signal Time", f"{green_time} seconds")
        with cols[1]:
            st.metric("Red Signal Time", f"{red_time} seconds")
        with cols[2]:
            st.metric("Yellow Signal Time", f"{yellow_time} seconds")

        # Recommendation based on detections
        if ambulance_count > 0:
            st.warning("⚠️ Emergency vehicle detected! Traffic signals adjusted to prioritize passage.")
        elif vehicle_count > 50:
            st.info("Heavy traffic detected. Green signal time increased.")
        elif vehicle_count < 10:
            st.info("Light traffic detected. Green signal time reduced.")
        else:
            st.info("Normal traffic flow. Standard signal timings applied.")

        # Replace the simple simulation button with this enhanced section
        st.markdown("---")
        st.subheader("🎮 Traffic Simulation Environment")
        # Create a card-like container for the simulation
        simulation_container = st.container()
        with simulation_container:
            # Two-column layout
            sim_col1, sim_col2 = st.columns([1, 2])

            with sim_col1:
                # Add an illustrative image or GIF
                st.image("maja-djokic-park-2-gif.gif",
                         caption="Simulation Preview", width=200)

            with sim_col2:
                st.markdown("### Interactive Traffic Simulator")
                st.markdown("""
                Run our real-time traffic simulator to visualize traffic flow with the 
                current signal timings. The simulation will demonstrate:
                - Vehicle movement through the intersection
                - Traffic signal cycling
                - Queue formation and clearance
                """)

                # Add buttons with icons
                if st.button("▶️ Launch Simulation", use_container_width=True):
                    st.info("Launching traffic simulation in a separate window...")
                    # Run the simulation as a subprocess
                    subprocess.Popen(["python", "simulation.py"])




