import cv2
import numpy as np 
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.0.1/bin/tesseract'


class TubeGame():
    
    def __init__(self, img):
        self.img = img
        self.phone, self.phone_bbox = self.getPhone()

        if self.phone.shape[1] > self.phone.shape[0]:
            self.phone =  cv2.rotate(self.phone, cv2.ROTATE_90_CLOCKWISE) 

        self.level, self.level_bbox = self.getLevel()
        
        if self.level_bbox[3][1] > self.phone.shape[0]/2:
            self.phone = cv2.rotate(self.phone, cv2.ROTATE_180)

        if self.level[0] == 'LEVEL': # Found the level
            print(f"FOUND {self.level[0]}: {self.level[1]}")
            self.phone = self.phone[self.level_bbox[3][1]: self.phone.shape[0], 0: self.phone.shape[1]]
            self.tubes, self.tubes_img = self.getTube()
            print(f"FOUND TUBES: {len(self.tubes)}")

    def __getitem__(self, index):
        return 

    def __len__(self):
        return len(self.tubes)

    def displayImg(self):
        cv2.imshow('Tube', self.img.copy())
        cv2.waitKey()

    def getThreshold(self, img, thresh_params = [81, 8]):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray, (5,5), 3)
        thresh = cv2.adaptiveThreshold(gauss, 255,
	        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_params[0], thresh_params[1])
        return thresh

    def getMask(self, img, l_range, h_range):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.inRange(hsv, l_range, h_range)

    def displayThreshold(self):
        thresh = self.getThreshold()
        cv2.imshow('Threshold', thresh)
        cv2.waitKey()

    # Format Box Coordinates in Clockwise Direction
    def clkwBox(self, box):
        ysort = sorted(box, key=lambda x: (x[1]))
        if ysort[0][0] > ysort[1][0]:
            ysort[0], ysort[1] = ysort[1], ysort[0]
        if ysort[2][0] < ysort[3][0]:
            ysort[2], ysort[3] = ysort[3], ysort[2]
        return np.array(ysort)

    def four_point_transform(self, image, box):
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

    # Get Bounding Box of Phone-Like Object
    def getPhone(self):
        thresh = self.getThreshold(self.img.copy())
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours =  sorted(contours[0], key=cv2.contourArea, reverse= True)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = self.clkwBox(np.int0(box))
        # box[:, 0][:2] += padding; box[:, 0][2:] -= padding
        # box[:, 1][:2] += padding; box[:, 1][2:] -= padding
        return self.four_point_transform(self.img.copy(), box), box

    def displayPhone(self):
        cv2.imshow('Tube', self.phone)
        cv2.waitKey()

    def getLevel(self):
        l_range = np.array([200, 200, 200])
        h_range = np.array([255, 255, 255])
        masked = cv2.bitwise_and(self.phone, self.phone, mask = self.getMask(self.phone, l_range, h_range))

        masked = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
        dilate = cv2.dilate(masked, kernel, iterations=8)   

        contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rects = []; MAX_AREA = (self.phone.shape[0] * self.phone.shape[1]) / 200 #

        for cnt in contours[0]:
            p = cv2.arcLength(cnt, True)
            epsilon = 0.01 * p 
            poly = cv2.approxPolyDP(cnt, epsilon, True)
            rect = list(cv2.boundingRect(poly))
            if rect[2] * rect[3] > MAX_AREA and rect[2] < 0.8 * self.phone.shape[0]:
                rect = [[rect[0], rect[1]], [rect[0] + rect[2], rect[1]], 
                        [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]]
                rects.append(rect)

        for item in np.int0(rects):
            img = self.four_point_transform(self.phone, item)
            
            text = pytesseract.image_to_string(img)
            if text[:5] == 'LEVEL':
                return text.split(), item
        return None, []

    def getTube(self, area_threshold = 0.8):
        thresh = self.getThreshold(self.phone, thresh_params = [5, 5])
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

        rects = np.array(sorted(rects, key = lambda x: (x[1], x[0]))); 
        MAX_AREA = max(rects[:, 2]) * max(rects[:, 3])
        

        for rect in rects:
            if rect[2] * rect[3] > area_threshold * MAX_AREA:
                rect = [[rect[0], rect[1]], [rect[0] + rect[2], rect[1]], 
                        [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]]
                tubes.append(rect)
                tubes_img.append(self.four_point_transform(self.phone, rect))

        return np.array(tubes), tubes_img

    def getGame(self, padding = 25):
        x_min = np.amin(self.tubes[:, :,0]) - padding
        x_max = np.amax(self.tubes[:, :,0]) + padding
        y_min = np.amin(self.tubes[:, :,1]) - padding
        y_max = np.amax(self.tubes[:, :,1]) + padding
        box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        return box

    def displayGame(self):
        box = self.getGame()
        cv2.imshow('game', self.four_point_transform(self.phone, box))
        cv2.waitKey()

    def displayTube(self, index: int):
        cv2.imshow('Tube ' + str(int), self.tubes_img[index])
        cv2.waitKey()


        


    


    

    





        

    



        
 