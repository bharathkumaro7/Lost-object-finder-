# Lost Object Finder

A web application that helps you find lost objects in images using deep learning and computer vision.

## Features

- 🔍 Upload a template image of your lost object
- 🏠 Upload a scene image (room, desk, etc.) where the object might be
- 🤖 Uses CNN-based visual similarity search to locate your object
- 🎯 Highlights the found object with a bounding box

## Tech Stack

- **Backend:** Flask (Python)
- **Computer Vision:** OpenCV, PyTorch, ResNet18
- **Frontend:** HTML, CSS (Modern UI)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/lost-object-finder.git
cd lost-object-finder
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and go to `http://127.0.0.1:5000`

## Usage

1. **Template Image:** Upload a clear photo of the object you're looking for (cropped tightly around the object)
2. **Scene Image:** Upload a photo of the area where you want to search
3. Click **Find My Object** and wait for the result

## Tips for Best Results

- Crop the template image to show only the object
- Use clear, well-lit images
- Objects with distinctive colors/patterns work best
- Avoid blurry or low-resolution photos

## License

MIT License
