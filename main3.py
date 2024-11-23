import cv2
import tkinter as tk
from tkinter import Button, Label, filedialog
from threading import Thread

# Yuzni aniqlash funksiyasi
def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [10, 4, 117, 123])
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:  # 70% dan yuqori ishonch darajasi
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bbox.append([x1, y1, x2, y2])
            # Yuz atrofida to‘g‘ri burchak chizish
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bbox

# Yosh va jinsni aniqlash funksiyasi
def process_age_gender(frame):
    global ageNet, genderNet, MODEL_MEAN_VALUES, ageList, genderList
    bboxs = faceBox(faceNet, frame)[1]
    results = []
    for bbox in bboxs:
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Jinsni aniqlash
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        # Yoshni aniqlash
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        # Jins va yoshni birlashtirib matn yaratish
        label = "{}, {}".format(gender, age)
        results.append(label)
    return results

# Video oqimini ishga tushirish
def video_stream():
    global running
    video = cv2.VideoCapture(0)
    while running:
        ret, frame = video.read()
        if not ret:
            break

        # Yosh va jins uchun kadrni qayta ishlash
        labels = process_age_gender(frame)
        for label in labels:
            # Kadrga matn joylashtirish
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Yoshni Aniqlash", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Fayl yuklash funksiyasi
def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_file(file_path)

# Faylni qayta ishlash funksiyasi
def process_file(file_path):
    global running
    running = True
    if file_path.endswith('.mp4') or file_path.endswith('.avi'):
        video = cv2.VideoCapture(file_path)
        while running:
            ret, frame = video.read()
            if not ret:
                break
            
            # Yosh va jins uchun kadrni qayta ishlash
            labels = process_age_gender(frame)
            for label in labels:
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Yoshni Aniqlash - Video", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        video.release()
    else:
        image = cv2.imread(file_path)
        if image is not None:
            labels = process_age_gender(image)
            for label in labels:
                cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Yoshni Aniqlash - Rasm", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Video oqimini boshlash
def start_video():
    global running
    running = True
    Thread(target=video_stream).start()

# Video oqimini to'xtatish
def stop_video():
    global running
    running = False

# Kamerani ko'rsatish oynasini ochish
def open_camera_window():
    main_window.withdraw()  # Asosiy oynani yashirish
    camera_window.deiconify()  # Kamera oynasini ko'rsatish

# Asosiy oynani ko'rsatish
def open_main_window():
    camera_window.withdraw()  # Kamera oynasini yashirish
    main_window.deiconify()  # Asosiy oynani ko'rsatish

# Ilovadan chiqish
def exit_app():
    global running
    running = False
    main_window.quit()

# Modellarni yuklash
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Neyron tarmoqlarni yuklash
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-19)', '(20-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Erkak', 'Ayol']

# Asosiy ilova oynasini tayyorlash
main_window = tk.Tk()
main_window.title("Yoshni Aniqlash")

# Sarlavha yorlig'ini yaratish
Label(main_window, text="Yoshni Aniqlash", font=("Helvetica", 16)).pack(pady=10)

# Kamera va fayl yuklash tugmalari
Button(main_window, text="Rasm yoki Video Yuklash", command=upload_file).pack(pady=10)
Button(main_window, text="Kamera", command=open_camera_window).pack(pady=10)
Button(main_window, text="Chiqish", command=exit_app).pack(pady=10)

# Kamera oynasini tayyorlash
camera_window = tk.Toplevel(main_window)
camera_window.title("Kamera")
camera_window.geometry("300x200")
camera_window.withdraw()  # Yashirin boshlash

# Kamera oynasidagi tugmalar
Button(camera_window, text="Yoshni Aniqlashni Boshlash", command=start_video).pack(pady=10)
Button(camera_window, text="Yoshni Aniqlashni To'xtatish", command=stop_video).pack(pady=10)
Button(camera_window, text="Bosh Oyna", command=open_main_window).pack(pady=10)

# GUI asosiy siklini ishga tushirish
main_window.mainloop()
