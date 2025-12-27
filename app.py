from flask import Flask, render_template, request, jsonify
from model_utils import predict_image
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        if file:
            img_bytes = file.read()
            label, confidence, status = predict_image(img_bytes)
            
            # Encode image to base64 for display
            image_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Detect image format from filename
            image_format = file.filename.split('.')[-1].lower()
            if image_format in ['jpg', 'jpeg']:
                image_format = 'jpeg'
            elif image_format not in ['png', 'gif', 'webp']:
                image_format = 'jpeg'  # default
            
            return render_template('index.html', 
                                   result=label, 
                                   confidence=confidence, 
                                   status=status,
                                   image_data=image_base64,
                                   image_format=image_format)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)