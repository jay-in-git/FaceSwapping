import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_mask(img, option='face', ratio=0.85):
    face_landmarks_list = face_recognition.face_landmarks(img)

    if not face_landmarks_list:
        print('[Error]: Please use human\'s image!')
        exit()
    
    mask = np.zeros((img.shape[0], img.shape[1]))
    
    if option == 'head':
        up_pos = [np.inf, -1]
        bottom_pos = [-np.inf, -1]
        left_pos = [-1, np.inf]
        right_pos = [-1, -np.inf]
        center = [0, 0]
        left_eye_pos = np.array([0, 0])
        right_eye_pos = np.array([0, 0])
        nose_pos = np.array([0, 0])
        lip_pos = np.array([0, 0])
        contours = []

        for key in face_landmarks_list[0].keys():
            for (j, i) in face_landmarks_list[0][key]:
                if i > bottom_pos[0]:
                    bottom_pos = [i, j]
                if i < up_pos[0]:
                    up_pos = [i, j]
                if j > right_pos[1]:
                    right_pos = [i, j]
                if j < left_pos[1]:
                    left_pos = [i, j]
                
                if key == 'left_eye':
                    left_eye_pos[0] += i
                    left_eye_pos[1] += j
                if key == 'right_eye':
                    right_eye_pos[0] += i
                    right_eye_pos[1] += j
                if key == 'nose_tip':
                    nose_pos[0] += i
                    nose_pos[1] += j
                if key == 'bottom_lip':
                    lip_pos[0] += i
                    lip_pos[1] += j

        

        left_eye_pos //= len(face_landmarks_list[0]['left_eye'])
        right_eye_pos //= len(face_landmarks_list[0]['right_eye'])
        nose_pos //= len(face_landmarks_list[0]['nose_tip'])
        lip_pos //= len(face_landmarks_list[0]['bottom_lip'])
        
        for key in face_landmarks_list[0].keys():
            for (j, i) in face_landmarks_list[0][key]:
                if 'eyebrow' in key:
                    x = round(1.1 * (i - nose_pos[0]) + nose_pos[0])
                    y = round(1.1 * (j - nose_pos[1]) + nose_pos[1])
                    contours.append([y,x])
                if 'chin' in key:
                    x = round(0.85 * (i - nose_pos[0]) + nose_pos[0])
                    y = round(0.85 * (j - nose_pos[1]) + nose_pos[1])
                    contours = [[y,x]] + contours

        contours = np.array(contours)
        center[0] = round((left_eye_pos[0] + right_eye_pos[0]) // 2)
        center[1] = (left_pos[1] + right_pos[1]) // 2
        vertex = round(1.05 * (bottom_pos[0]) - center[0])
        covertex = round(1.3 * (right_pos[1] - center[1]))

        cv2.ellipse(mask, (center[1], center[0]), (covertex, vertex), 0, 0, 360, 1, -1)
        
        for t in range(3):
            face_landmarks_list[0]['chin'] = sorted(face_landmarks_list[0]['chin'], key = lambda x:(x[0]))
            for i in range(1, len(face_landmarks_list[0]['chin'])):
                x = (face_landmarks_list[0]['chin'][i - 1][0] + face_landmarks_list[0]['chin'][i][0]) // 2
                y = (face_landmarks_list[0]['chin'][i - 1][1] + face_landmarks_list[0]['chin'][i][1]) // 2
                face_landmarks_list[0]['chin'].append((x, y))
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j]:
                    if j < bottom_pos[1]:
                        for (j_, i_) in face_landmarks_list[0]['chin']:
                            if i > i_ and j < j_ and j_ < bottom_pos[1]:
                                mask[i][j] = 0
                                break
                    else:
                        for (j_, i_) in face_landmarks_list[0]['chin']:
                            if i > i_ and j > j_ and j_ > bottom_pos[1]:
                                mask[i][j] = 0
                                break
        return mask
        
    elif option == 'face':
        nose_pos = np.array([0, 0])
        contours = []

        for (j, i) in face_landmarks_list[0]['nose_tip']:
            nose_pos[0] += i
            nose_pos[1] += j
        
        nose_pos //= len(face_landmarks_list[0]['nose_tip'])
        
        for (j, i) in face_landmarks_list[0]['left_eyebrow']:
            x = round(1.3 * ratio * (i - nose_pos[0]) + nose_pos[0])
            y = round(1.3 * ratio * (j - nose_pos[1]) + nose_pos[1])
            contours.append([y,x])
        
        for (j, i) in face_landmarks_list[0]['right_eyebrow']:
            x = round(1.3 * ratio * (i - nose_pos[0]) + nose_pos[0])
            y = round(1.3 * ratio * (j - nose_pos[1]) + nose_pos[1])
            contours.append([y,x])

        for (j, i) in face_landmarks_list[0]['chin']:
            x = round(ratio * (i - nose_pos[0]) + nose_pos[0])
            y = round(ratio * (j - nose_pos[1]) + nose_pos[1])
            contours = [[y,x]] + contours
        
        contours = np.array(contours)
        cv2.drawContours(mask, [contours.reshape(-1,1,2)], -1, 1, -1)
        return mask
    
    else:
        option2key = {'eye' : 'eye', 'mouth' : 'lip', 'nose' : 'nose'}
        up_pos = [np.inf, -1]
        bottom_pos = [-np.inf, -1]
        left_pos = [-1, np.inf]
        right_pos = [-1, -np.inf]

        for key in face_landmarks_list[0].keys():
            if option2key[option] in key:
                for (j, i) in face_landmarks_list[0][key]:
                    if i > bottom_pos[0]:
                        bottom_pos = [i, j]
                    if i < up_pos[0]:
                        up_pos = [i, j]
                    if j > right_pos[1]:
                        right_pos = [i, j]
                    if j < left_pos[1]:
                        left_pos = [i, j]
        if option == 'eye':
            cv2.rectangle(mask, (left_pos[1], up_pos[0]), (right_pos[1], round(1.05*bottom_pos[0])), 1, -1)
        else:
            cv2.rectangle(mask, (left_pos[1], up_pos[0]), (right_pos[1], bottom_pos[0]), 1, -1)

        return mask
    

if __name__ == '__main__':
    img = cv2.imread('./elon.jpeg')
    mask = get_mask(img, 'face')
    cv2.imwrite("mask.jpg", mask)
