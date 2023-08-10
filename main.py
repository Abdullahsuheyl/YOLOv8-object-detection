from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog


class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
    7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
    28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

model = YOLO("yolov8n.pt")

def mainPage():
    page = tk.Tk()
    page.title("Goruntu")
    page.geometry("500x500")
    
    def open_and_display_image():
        imagePath = openImage()  
        if imagePath:
            displayImage(imagePath)  
            
    imageInput = tk.Button(page, text="Görseli Seç", command=open_and_display_image,
                           width=120,height=120)
    imageInput.pack()
    
    page.mainloop()

def openImage():
    imagePath = filedialog.askopenfilename(filetypes=[("Image files", 
                                                        "*.png *.jpg *.jpeg *.gif")])
    return imagePath

def displayImage(imagePath):
    image = cv2.imread(imagePath)
    detections = model(image)[0]
    
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, ID = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        color = (0, 255, 0)  # Yeşil renk
        thickness = 2
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        if int(ID) in class_names:
            class_name = class_names[int(ID)]
        else:
            class_name = 'Unknown'
            
        
        text = f"{class_name},{score:.2f}"
        text_position = (x1+50, y1) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)  
        font_thickness = 1
        
        
        cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness)
        
    cv2.imshow("Goruntu", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

mainPage()




