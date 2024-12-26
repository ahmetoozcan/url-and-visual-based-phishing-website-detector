import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
import networkx as nx
from url_extractor import extract_features

@tf.keras.utils.register_keras_serializable()
class GNN(Model):
    def __init__(self, num_features = 41, num_classes = 2, name='GNN', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_features = num_features
        self.num_classes = num_classes
        self.dense1 = layers.Dense(64, activation='relu', name='dense1')
        self.dropout1 = layers.Dropout(0.2, name='dropout1')    
        self.dense2 = layers.Dense(32, activation='relu', name='dense2')
        self.dropout2 = layers.Dropout(0.2, name='dropout2')
        self.dense3 = layers.Dense(16, activation='relu', name='dense3')
        self.dropout3 = layers.Dropout(0.5, name='dropout3')
        self.output_layer = layers.Dense(num_classes, activation='softmax', name='output_layer')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.dropout3(x)
        return self.output_layer(x)
    
    def get_config(self):
        config = super(GNN, self).get_config()
        config.update({
            'num_features': self.num_features,
            'num_classes': self.num_classes,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ModelPreprocessor:
    @staticmethod
    def preprocess_gnn(features_df,scaler):
        # Normalize features
        normalized_features = scaler.transform(features_df)
        
        # Create dataset
        dataset = pd.DataFrame(normalized_features, columns=features_df.columns)
        
        # Create graph
        G = nx.Graph()
        for index, row in dataset.iterrows():
            G.add_node(index, features=row.values)
        
        # Extract node features
        node_features = np.array([G.nodes[node]['features'] for node in G.nodes])
        return node_features.reshape(1, -1)

    @staticmethod
    def preprocess_cnn(features_df, scaler):
        normalized_features = scaler.transform(features_df)
        return normalized_features.reshape(normalized_features.shape[0], normalized_features.shape[1], 1)

    @staticmethod
    def preprocess_rnn_lstm(features_df, scaler):
        normalized_features = scaler.transform(features_df)
        return normalized_features.reshape((normalized_features.shape[0], 1, normalized_features.shape[1]))

    @staticmethod
    def preprocess_fnn(features_df, scaler):
        return scaler.transform(features_df)


# Update the load_image function
def load_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to load image
def load_image_button():
    global img_array
    image_path = filedialog.askopenfilename()
    if not image_path:
        return
    img_array = load_image(image_path)
    img = Image.open(image_path)
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Function to predict
def predict():
    # Display result
    result_label.config(text=f"Result: Loading...")
    result_label_url.config(text=f"Result URL: Loading...")
    result_label_visual.config(text=f"Result Visual: Loading...")
    root.update()

    url = url_entry.get()
    if img_array is None:
        result_label.config(text="Please load an image first.")
        return

    # Extract features from URL
    url_features = extract_features(url)
    url_features_df = pd.DataFrame([url_features])

    # Load the scaler
    scaler = joblib.load('normalization/url.pkl')

    # Load the selected URL model and preprocess features accordingly
    selected_model = model_var.get()
    
    if selected_model == 'GNN':
        normalized_features = ModelPreprocessor.preprocess_gnn(url_features_df,scaler)
        url_model = load_model('url_models/gnn_model.keras', custom_objects={'GNN': GNN})
    elif selected_model == 'CNN':
        normalized_features = ModelPreprocessor.preprocess_cnn(url_features_df, scaler)
        url_model = load_model('url_models/cnn_url_model.keras')
    elif selected_model in ['LSTM', 'RNN']:
        normalized_features = ModelPreprocessor.preprocess_rnn_lstm(url_features_df, scaler)
        model_path = f'url_models/{selected_model.lower()}_model.keras'
        url_model = load_model(model_path)
    elif selected_model == 'FNN':
        normalized_features = ModelPreprocessor.preprocess_fnn(url_features_df, scaler)
        url_model = load_model('url_models/fnn_model.keras')
    else:
        result_label.config(text="Please select a URL model.")
        return

    # Predict with URL model
    url_prediction = url_model.predict(normalized_features)
    if selected_model == 'CNN':
        url_prediction = np.round(url_prediction).flatten()
    else:
        url_prediction = np.round(np.argmax(url_prediction, axis=1))

    # Predict with visual model
    visual_prediction = visual_model.predict(img_array)
    visual_prediction = np.round(visual_prediction).astype(int)

    # Combine predictions based on selected gate
    selected_gate = gate_var.get()
    if selected_gate == 'AND':
        final_prediction = 1 if url_prediction[0] == 1 and visual_prediction[0][0] == 1 else 0
    elif selected_gate == 'OR':
        final_prediction = 1 if url_prediction[0] == 1 or visual_prediction[0][0] == 1 else 0
    else:
        result_label.config(text="Please select a gate option.")
        return

    # Display result
    result_label_url.config(text=f"Result URL: {'Phishing' if url_prediction[0] == 1 else 'Legitimate'}")
    result_label_visual.config(text=f"Result Visual: {'Phishing' if visual_prediction[0][0] == 1 else 'Legitimate'}")
    result_label.config(text=f"Result: {'Phishing' if final_prediction == 1 else 'Legitimate'}")


# Create the main window
root = tk.Tk()
root.title("Phishing Detection")
root.geometry("600x600")

# Create and place the URL entry
tk.Label(root, text="URL:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
url_entry = tk.Entry(root, width=50)
url_entry.grid(row=0, column=1, padx=10, pady=10, sticky='ew')

# Replace the URL model selection combobox with radio buttons
model_var = tk.StringVar()
tk.Label(root, text="URL Model:").grid(row=1, column=0, padx=10, pady=10, sticky='w')

# Create frame to hold model radio buttons horizontally
model_frame = ttk.Frame(root)
model_frame.grid(row=1, column=1, sticky='ew')
models = ['LSTM', 'GNN', 'CNN', 'FNN', 'RNN']
for i, model in enumerate(models):
    ttk.Radiobutton(model_frame, text=model, variable=model_var, value=model).grid(row=0, column=i, padx=5)

# Create frame to hold gate radio buttons horizontally with less spacing
gate_frame = ttk.Frame(root)
gate_frame.grid(row=2, column=1, sticky='w')
gate_var = tk.StringVar()
tk.Label(root, text="Gate:").grid(row=2, column=0, padx=10, pady=10, sticky='w')
ttk.Radiobutton(gate_frame, text="AND", variable=gate_var, value='AND').grid(row=0, column=0, padx=5)
ttk.Radiobutton(gate_frame, text="OR", variable=gate_var, value='OR').grid(row=0, column=1, padx=5)

# Create and place the load image button
load_image_button = tk.Button(root, text="Load Image", command=load_image_button)
load_image_button.grid(row=3, column=0, columnspan=2, pady=10, sticky='ew')

# Create and place the predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=4, column=0, columnspan=2, pady=10, sticky='ew')

# Create and place the image label
image_label = tk.Label(root)
image_label.grid(row=5, column=0, columnspan=2, pady=10, sticky='ew')

# Create and place the result label
result_label_url = tk.Label(root, text="Result URL: ")
result_label_url.grid(row=6, column=0, columnspan=2, pady=10, sticky='ew')
# Create and place the result label
result_label_visual = tk.Label(root, text="Result Visual: ")
result_label_visual.grid(row=7, column=0, columnspan=2, pady=10, sticky='ew')
# Create and place the result label
result_label = tk.Label(root, text="Result: ")
result_label.grid(row=8, column=0, columnspan=2, pady=10, sticky='ew')

# Configure grid weights for resizing
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(5, weight=1)

# Load the visual model at the start
visual_model = load_model('visual_models/cnn_visual.keras')

# Run the application
root.mainloop()