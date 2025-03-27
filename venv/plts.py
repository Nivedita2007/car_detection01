import cv2
import matplotlib.pyplot as plt
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


mask = './mask_1920_1080.png'
video_path = './samples/parking_1920_1080_loop.mp4'

mask = cv2.imread(mask, 0)

cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None

frame_nmr = 0
ret = True
step = 30

while ret:
    ret, frame = cap.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            #spot_status = empty_or_not(spot_crop)  # Get spot status (empty or occupied)
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

            # Ensure spots_status is updated correctly
            #spots_status[spot_indx] = spot_status

        print([diffs[j] for j in np.argsort(diffs)][::-1])
        plt.figure()
        plt.hist([diffs[j] / np.amax(diffs) for j in np.argsort(diffs)][::-1])
        plt.show()
        if(frame_nmr==300):
            plt.show()

    if frame_nmr % step == 0:  # Update previous_frame
        previous_frame = frame.copy()

        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            # Get the spot status by calling empty_or_not
            spot_status = empty_or_not(spot_crop)

            # Now use the spot_status to draw the rectangle
            if spot_status:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # Green for available
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)  # Red for occupied

    # Handle the case where spots_status might have None values
    available_spots = sum(1 for status in spots_status if status is True)
    total_spots = len(spots_status)

    # Display the available spots count
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(available_spots), str(total_spots)), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_nmr+=1

cap.release()
cv2.destroyAllWindows()
