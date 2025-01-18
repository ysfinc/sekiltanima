import cv2
import numpy as np

def manual_arc_length(contour):
    length = 0
    for i in range(1, len(contour)):
        x1, y1 = contour[i - 1][0]  
        x2, y2 = contour[i][0]     
        length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  
    x1, y1 = contour[-1][0]
    x2, y2 = contour[0][0]
    length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    return length

def perpendicular_distance(point, line):
    x0, y0 = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

def douglas_peucker(point_list, epsilon):
    dmax = 0
    index = 0
    end = len(point_list)
    
    for i in range(1, end - 1):
        d = perpendicular_distance(point_list[i], (point_list[0], point_list[end - 1]))
        if d > dmax:
            index = i
            dmax = d
    
    result_list = []

    if dmax > epsilon:
        rec_results1 = douglas_peucker(point_list[:index + 1], epsilon)
        rec_results2 = douglas_peucker(point_list[index:], epsilon)
        result_list = rec_results1[:-1] + rec_results2
    else:
        result_list = [point_list[0], point_list[end - 1]]
    
    return result_list

font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.imread("sekiller.jpg")
cv2.namedWindow("IMG", cv2.WINDOW_NORMAL)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
blurred = cv2.GaussianBlur(threshold, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

kernel = np.ones((7, 7), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2)
edges = cv2.erode(edges, kernel, iterations=1)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

h = img.shape[0]
w = img.shape[1]
min_area = h * w * 0.01
max_area = h * w * 0.3

filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area <= area <= max_area:
        filtered_contours.append(contour)

mask = np.zeros_like(gray)

for cnt in filtered_contours:
    cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    alan = cv2.contourArea(cnt)
    
    length = manual_arc_length(cnt)
    epsilon = 0.025 * length
    
    approx_points = douglas_peucker(cnt.reshape(-1, 2), epsilon)
    approx = np.array(approx_points, dtype=np.int32).reshape((-1, 1, 2))

    cv2.drawContours(img, [approx], -1, (0, 0, 255), 2)

    x = approx.ravel()[0]
    y = approx.ravel()[1]

    x, y, w, h = cv2.boundingRect(approx)
    kenar_sayisi=len(approx)-1
    if kenar_sayisi == 3:
        cv2.putText(img, "Ucgen", (x, y - 20), font, 1, (0, 0, 0), 3)
    elif kenar_sayisi == 4:
        cv2.putText(img, "Dortgen", (x, y + 20), font, 1, (0, 0, 0), 3)
    elif kenar_sayisi == 5:
        cv2.putText(img, "Besgen", (x, y), font, 1, (0, 0, 0), 3)
    elif kenar_sayisi == 6:
        cv2.putText(img, "Altigen", (x, y), font, 1, (0, 0, 0), 3)
    else:
        cv2.putText(img, "Cember", (x, y), font, 1, (0, 0, 0), 3)

cv2.imshow("IMG", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
