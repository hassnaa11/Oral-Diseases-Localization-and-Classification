import io 
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image #python imaging library(Pillow), for image processing
import uvicorn #is a web server to run FastAPI framework
from torchvision import transforms, models

app=FastAPI(title="Oral_Disease_Image_Classifier_API",
              description="An API to classify oral disease images ",version="1.0")

#Device configuration
device='cuda' if torch.cuda.is_available() else 'cpu'

#same trasnorms used in main.py
IMAGENET_MEAN=[0.485, 0.456, 0.406]
IMAGENET_STD=[0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

#Load the model
num_classes=6
def load_model():
    model=models.convnext_small(weights=None)
    model.classifier[2]=nn.Linear(model.classifier[2].in_features, num_classes)

    #load trained weights
    stat_dict=torch.load('best_model.pth', map_location=device)
    model.load_state_dict(stat_dict)
    model.to(device)
    model.eval()
    return model
model =load_model()

classes=["Calculus", "Data caries", "Gingivitis", "Mouth Ulcer", 'Tooth Discoloration','hypodontia']

#Helper function to read image
def classify_image(image: Image.Image): #Image function uses PIL image
    img_tensor= test_transform(image).unsqueeze(0).to(device) #unsqueeze(0) adds a batch dimension 

    with torch.no_grad():        #no backpropagation needed during inference/predicting
        logits=model(img_tensor) #o/p is raw and unnormalized results(not probabilities yet)
        probs=torch.softmax(logits, dim=1).cpu().numpy()[0] #convert logits into probabilities that sum to 1 , and move result to cpu then remove batch dimension
    
    #index of best class
    predicted_index=probs.argmax()
    predicted_class=classes[predicted_index]

    #return class and probability
    return {
        "predicted_class": predicted_class,
        "probability": float(probs[predicted_index])
    }


#API endpoint to classify image
@app.get("/")
def home():
    return {"message": "Oral Disease API is running! Go to /docs to test."}

@app.post('/predict')
async def predict(file: UploadFile=File(...)):
    #load the uploaded file into PIL image
    image_bytes=await file.read()
    image=Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Run prediction
    result = classify_image(image)

    return {
        "filename": file.filename,
        "result": result
    }
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8050)


