import cv2
import numpy as np
from numpy.core.multiarray import empty 
import pytesseract
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import math

from skimage import io

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.0.1/bin/tesseract'

class Tubes:
    
    def __init__(self, img, img_type):
        types = ['phone', 'screenshot']
        self.__img = img
        if img_type == types[0]:
            self.__phone, self.__phone_bbox = self.__findPhone()

            if self.__phone.shape[1] > self.__phone.shape[0]:
                self.__phone =  cv2.rotate(self.__phone, cv2.ROTATE_90_CLOCKWISE) 
        elif img_type == types[1]:
            self.__phone = img
            self.__phone_bbox = [[0, 0], [img.shape[1], 0], 
            [0, img.shape[0]], [img.shape[1], img.shape[0]]]

        self.__level, self.__level_bbox = self.__findLevel()
        
        if self.__level_bbox[3][1] > self.__phone.shape[0]/2:
            self.__phone = cv2.rotate(self.__phone, cv2.ROTATE_180)

        if self.__level[0] == 'LEVEL': # Found the level
            print(f"FOUND {self.__level[0]}: {self.__level[1]}")
            self.__phone = self.__phone[self.__level_bbox[3][1]: self.__phone.shape[0], 0: self.__phone.shape[1]]
            self.__tubes, self.__tubes_img = self.__findTube()
            self.__gameType = self.__detectGameType()
            print(f"FOUND TUBES: {len(self.__tubes)}")
            self.__tube_colors, self.__colors = self.__findTubeColors()
            if len(self.__colors) == len(self.__tubes) - 2:
                print(f"TUBE COLORS FOUND")
            else:
                print(f"COLORS FOUND INCORRECTLY")
        else:
            print("LEVEL AND GAME NOT FOUND")

            
    def __finditem__(self, index):
        return self.__tube_colors[index]

    def __len__(self):
        return len(self.__tubes)

    def setTubeColors(self, tube, index: int):
        self.__tube_colors[index] = tube

    def getTubeColors(self):
        return self.__tube_colors

    def getGameColors(self):
        return self.__colors

    def displayImg(self):
        cv2.imshow('Tube', self.__img.copy())
        cv2.waitKey()

    def __findThreshold(self, img, thresh_params = [81, 8]):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray, (5,5), 3)
        thresh = cv2.adaptiveThreshold(gauss, 255,
	        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_params[0], thresh_params[1])
        return thresh

    def __findMask(self, img, l_range, h_range, color = cv2.COLOR_BGR2RGB):
        hsv = cv2.cvtColor(img, color)
        return cv2.inRange(hsv, l_range, h_range)

    def displayThreshold(self):
        thresh = self.__findThreshold()
        cv2.imshow('Threshold', thresh)
        cv2.waitKey()

    # Format Box Coordinates in Clockwise Direction
    def __clkwBox(self, box):
        ysort = sorted(box, key=lambda x: (x[1]))
        if ysort[0][0] > ysort[1][0]:
            ysort[0], ysort[1] = ysort[1], ysort[0]
        if ysort[2][0] < ysort[3][0]:
            ysort[2], ysort[3] = ysort[3], ysort[2]
        return np.array(ysort)

    def __four_point_transform(self, image, box):
        (tl, tr, br, bl) = box

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(np.float32(box), dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    # find Bounding Box of Phone-Like Object
    def __findPhone(self):
        thresh = self.__findThreshold(self.__img.copy())
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours =  sorted(contours[0], key=cv2.contourArea, reverse= True)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = self.__clkwBox(np.int0(box))
        return self.__four_point_transform(self.__img.copy(), box), box

    def displayPhone(self):
        cv2.imshow('Tube', self.__phone)
        cv2.waitKey()

    def __findLevel(self):
        l_range = np.array([200, 200, 200])
        h_range = np.array([255, 255, 255])
        masked = cv2.bitwise_and(self.__phone, self.__phone, mask = self.__findMask(self.__phone, l_range, h_range))

        masked = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
        dilate = cv2.dilate(masked, kernel, iterations=8)   

        contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rects = []; MAX_AREA = (self.__phone.shape[0] * self.__phone.shape[1]) / 200 #

        for cnt in contours[0]:
            p = cv2.arcLength(cnt, True)
            epsilon = 0.01 * p 
            poly = cv2.approxPolyDP(cnt, epsilon, True)
            rect = list(cv2.boundingRect(poly))
            if rect[2] * rect[3] > MAX_AREA and rect[2] < 0.8 * self.__phone.shape[0]:
                rect = [[rect[0], rect[1]], [rect[0] + rect[2], rect[1]], 
                        [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]]
                rects.append(rect)

        for item in np.int0(rects):
            img = self.__four_point_transform(self.__phone, item)
            
            text = pytesseract.image_to_string(img)
            if text[:5] == 'LEVEL':
                return text.split(), item
        return None, []

    # Tubes are from Top to Bottom, Left to Right
    def __findTube(self, area_threshold = 0.8):
        thresh = self.__findThreshold(self.__phone, thresh_params = [5, 5])
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rects = []; tubes = []; tubes_img = []

        for cnt in contours[0]:
            p = cv2.arcLength(cnt, True)
            epsilon = 0.01 * p 
            poly = cv2.approxPolyDP(cnt, epsilon, True)
            rect = list(cv2.boundingRect(poly))
            rect = np.array(rect)
            if rect[3] > rect[2]:
                rects.append(rect)

        rects = np.array(sorted(rects, key = lambda x: (x[0], x[1])))
        MAX_AREA = max(rects[:, 2]) * max(rects[:, 3])

        
        for rect in rects:
            if rect[2] * rect[3] > area_threshold * MAX_AREA:

                rect[0] += rect[2]/8; rect[2] -= 2 * rect[2]/8
                rect[1] += rect[3]/6; rect[3] -= rect[3]/6

                rect = [[rect[0], rect[1]], [rect[0] + rect[2], rect[1]], 
                        [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]]
                tubes.append(rect)
                tubes_img.append(self.__four_point_transform(self.__phone, rect))

        return np.array(tubes), tubes_img

    def __findGame(self, padding = 25):
        x_min = np.amin(self.__tubes[:, :,0]) - 2 * padding
        x_max = np.amax(self.__tubes[:, :,0]) + 2 * padding
        y_min = np.amin(self.__tubes[:, :,1]) - 2 * padding
        y_max = np.amax(self.__tubes[:, :,1]) + 2 * padding
        box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        return box, self.__four_point_transform(self.__phone, box)

    def displayGame(self):
        box, game_img = self.__findGame()
        cv2.imshow('game', game_img)
        cv2.waitKey()

    def displayTube(self):
        for index, tube in enumerate(self.__tubes_img):
            cv2.imshow('Tube ' + str(index + 1), tube)
            cv2.waitKey()

    def plot_color(self, color):
        bar = np.zeros((50, 300, 3), dtype = "uint8")
        startX = 0
        endX = startX + (300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
        
        return bar

    def __detectGameType(self):
        level = int(self.__level[1])
        return 'mystery' if level % 5 == 0 else 'color'

    def __findTubeColors(self):
        colors = []
        gamecolors = []
        for tubes in self.__tubes_img:
            color = []
            height = math.floor(tubes.shape[0]/4 - 1)
            width = math.floor(tubes.shape[1]/4 - 1)
            y_top = tubes.shape[0] - height
            y_bot = tubes.shape[0]
            h_pad = math.floor(width / 3)
            v_pad = math.floor(height / 5)
            box_index = 0
            while y_top > 0:
                color_img = tubes[y_top + v_pad: y_bot - 3 * v_pad,  2 * h_pad: tubes.shape[1] - 2 * h_pad]

                y_top -= height; y_bot -= height
                
                color_img = color_img.reshape((color_img.shape[0] * color_img.shape[1], 3))
                clt = MiniBatchKMeans(n_clusters = 1)
                clt.fit(color_img)

                boxcolor = clt.cluster_centers_[0]
                

                if self.__gameType == 'mystery' and box_index in range(3):
                    if not all(boxcolor < 50):
                        color.append(-1)
                        box_index += 1
                elif len(gamecolors) == 0:
                    gamecolors.append(boxcolor)
                    color.append(1)
                elif all(boxcolor < 50):
                    continue
                else:
                    found = False
                    min = []
                    for index, c in enumerate(gamecolors):
                        if (self.__rgb_euclid(boxcolor, c) < 50):
                            min.append([self.__rgb_euclid(boxcolor, c), index])
                            found = True

                    if not found:
                        gamecolors.append(boxcolor)
                        color.append(len(gamecolors))
                    else:
                        color.append(min[np.argmin(min, axis = 0)[0]][1] + 1)
            box_index = 0
            colors.append(color)
            
        # ----------- Swap even and odd tubes ----------- #
        #      Not working for one or 3 rows of tubes
        
        #odd = []; even = []
        #for ind, clr in enumerate(colors):
        #    if ind % 2 == 0:
        #        even.append(clr)
        #    else:
        #        odd.append(clr)

        #return even + odd, gamecolors  
        return colors, gamecolors
                                 
    def __rgb_euclid(self, color1, color2):
       diff = np.array(color2) - np.array(color1)
       return math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)

    
    def plot_tubes(self, tubes, step = []):
        tubes_img = np.zeros((350, len(self.__tubes) * 65, 3), dtype = "uint8")
        startX = 20
        startY = 240
        percents = [0.25 for i in range(4)]

        for index, tube in enumerate(tubes):
            colors = []
            for j in range(4 - len(tube)):
                tube = np.append(tube, 0)
            for i in tube:
                if i == 0:
                    colors.append(np.array([255, 255, 255]))
                elif i == -1:
                    colors.append(np.array([204, 153, 255]))
                else:
                    colors.append(self.__colors[i-1])
            endX = startX + 40
            for (percent, color) in zip(percents, colors):
                endY = startY - (percent * 200)
                cv2.rectangle(tubes_img, (int(startX), int(startY)), (int(endX), int(endY)),
                    color.astype("uint8").tolist(), -1)
                cv2.rectangle(tubes_img, (int(startX), int(startY)), (int(endX), int(endY)),
                    [255, 255, 255], 3)
                if len(step) != 0:
                    if index == step[0]:
                        cv2.rectangle(tubes_img, (int(startX) + 10, 280), (int(endX) - 10, 320),
                        [52, 64, 235], -1)
                    elif index == step[1]:
                        cv2.rectangle(tubes_img, (int(startX) + 10, 280), (int(endX) - 10, 320),
                        [52, 235, 58], -1)

                startY = endY
            startX = endX + 20
            startY = 240
        cv2.imshow('Detected Tubes', tubes_img)
        cv2.waitKey()