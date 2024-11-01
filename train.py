from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
import pillow_avif

def main():
    model = YOLO('yolov8n.pt')
    try:
        model.to('cuda')
    except:
        pass
    
    print("loaded model")
    img = Image.open("./content/img.png")
    img.save("./content/img.png")
    img = cv2.imread("./content/img.png")
    #print("./content/img.png")
    #img = img[..., ::-1]
    #print("loaded image", img)
    results = model.train(data="data.yaml", epochs=200, imgsz=640, amp=False, resume=False)
    results = model.predict(img, conf=.1)
    
    for r in results:
        annotator = Annotator(img)
        
        boxes = r.boxes
        for box in boxes:
            b = tuple(map(int, box.xyxy[0]))  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            img = cv2.putText(img, model.names[int(c)] + " " + str(round(float(box.conf[0]), 5)), (b[0] - 10,b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 10, cv2.LINE_AA)
            img = cv2.putText(img, model.names[int(c)] + " " + str(round(float(box.conf[0]), 5)), (b[0] - 10,b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5, cv2.LINE_AA)
            
    cv2.imshow("Yolo test:", cv2.resize(img, (600, int(600 * img.shape[0] / img.shape[1]))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./content/imgRes.png", img)

if __name__ == '__main__':
    main()
