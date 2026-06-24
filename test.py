import argparse
import os
from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Layer, Input, Dense, LSTM, Embedding, Dropout, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to the test image")
args = vars(ap.parse_args())
img_path = args['image']

# 2. Re-create the Attention Layer for loading
class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

# 3. Extract Spatial Grid Features (100, 2048)
def extract_features(filename, model):
    try:
        image = Image.open(filename).convert('RGB')
    except Exception as e:
        print(f"ERROR: Couldn't open image! Check path: {filename}")
        return None
        
    image = image.resize((299, 299))
    image = np.array(image)
    image = np.expand_dims(image, axis=0) / 127.5 - 1.0
    
    # Extract shape: (1, 10, 10, 2048)
    spatial_feat = model.predict(image, verbose=0)
    # Reshape to match the trained model's input token dimension
    return np.reshape(spatial_feat, (100, 2048))

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# 4. Generate description string with Post-Padding
def generate_desc(model, tokenizer, photo, max_length):
    in_text = '<start>'
    photo = np.expand_dims(photo, axis=0) # Add batch dimension -> (1, 100, 2048)
    
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])
        sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
        
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end' or word == '<end>':
            break
            
    final_caption = in_text.replace('<start>', '').replace('<end>', '').replace('end', '').strip()
    return final_caption

# 5. Initialization
max_length = 32
tokenizer = load(open("tokenizer.p", "rb"))
vocab_size = len(tokenizer.word_index) + 1

# Update this to point to your new downloaded attention model weights/file
model_file_path = 'best_attention_model.keras' 

print("Loading Attention-Based Model...")
# CRITICAL: We pass custom_objects so Keras understands our custom Attention layer
model = tf.keras.models.load_model(
    model_file_path, 
    custom_objects={'BahdanauAttention': BahdanauAttention}
)

# Initialize raw Xception feature extractor (No pooling!)
print("Loading Feature Extractor...")
xception_model = Xception(include_top=False, weights='imagenet')

# 6. Run Inference Pipeline
print("Processing Image...")
photo = extract_features(img_path, xception_model)

if photo is not None:
    description = generate_desc(model, tokenizer, photo, max_length)
    print("\n" + "="*40)
    print(f"PREDICTED CAPTION: {description}")
    print("="*40 + "\n")
    
    # Display image on screen locally
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# python test.py --image Flickr8k_Dataset/Flicker8k_Dataset/44129946_9eeb385d77.jpg  two people sit on bench watching the water
# python test.py --image Flickr8k_Dataset/Flicker8k_Dataset/3711030008_3872d0b03f.jpg  woman in pink sweatshirt and pink cross to cross the street
# python test.py --image Flickr8k_Dataset/Flicker8k_Dataset/3710468717_c051d96a5f.jpg two children and dog running in snow
# correct