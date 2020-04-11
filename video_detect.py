import numpy as np
import imutils, time, cv2, os, logging

def check_time(path):
    model = "mask-rcnn-coco"

    labelsPath = os.path.sep.join([model, "object_detection_classes_coco.txt"])
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    weightsPath = os.path.sep.join([model, "frozen_inference_graph.pb"])
    configPath = os.path.sep.join([model, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

    logging.info("loading Mask R-CNN from disk...")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    vs = cv2.VideoCapture(path)

    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        logging.info("{} total frames in video".format(total))
    except:
        logging.info("could not determine # of frames in video")
        total = -1

    (grabbed, frame) = vs.read()

    if not grabbed: return

    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    boxes = net.forward(["detection_out_final"])[0]
    end = time.time()
    vs.release()
    return int(total*10*(end-start))

def detect_video(path):
    model = "mask-rcnn-coco"

    labelsPath = os.path.sep.join([model, "object_detection_classes_coco.txt"])
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    weightsPath = os.path.sep.join([model, "frozen_inference_graph.pb"])
    configPath = os.path.sep.join([model, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

    logging.info("loading Mask R-CNN from disk...")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    vs, flag = cv2.VideoCapture(path), False

    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        logging.info("{} total frames in video".format(total))
    except:
        logging.info("could not determine # of frames in video")
        total = -1
    found = set()
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break

        (grabbed, frame) = vs.read()

        if not grabbed:	break

        blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        boxes = net.forward(["detection_out_final"])[0]
        end = time.time()

        for i in range(0, boxes.shape[2]):
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]

            if confidence > 0.5:
                (H, W) = frame.shape[:2]
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                boxW, boxH = endX - startX, endY - startY

                color = COLORS[classID]
                color = [int(c) for c in color]
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                text = "{}: {:.4f}".format(LABELS[classID], confidence)
                cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                found.add(LABELS[classID])

        if not flag:
            flag = True
            if total > 0:
                elap = (end - start)
                logging.info("single frame took {:.4f} seconds".format(elap))
                logging.info("Estimated total time to finish: {:.4f} seconds".format(elap * total))
        cv2.putText(frame, "Press 'q' to QUIT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Frame", frame)
    
    if found: logging.info("Following are the objects detected in the video: \n {}".format('\n'.join(found)))
    else: logging.info("No Objects detected in the provided Video")

    logging.info("cleaning up...")
    vs.release()
    cv2.destroyAllWindows()
