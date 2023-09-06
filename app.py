from transformers import ViTFeatureExtractor, ViTForImageClassification
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load the model and tokenizer
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.route('/predict', methods=['POST'])
def predict():
   
    image_bytes = request.files.get('image').read()


    image = Image.open(io.BytesIO(image_bytes))


    inputs = feature_extractor(images=image, return_tensors="pt")
    

    outputs = model(**inputs)
    predicted_class = model.config.id2label[int(outputs.logits.argmax(-1))]

    return jsonify({"class": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
