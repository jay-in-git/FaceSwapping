import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_mask(img, option='face'):
    face_landmarks_list = face_recognition.face_landmarks(img)
    mask = np.zeros((img.shape[0], img.shape[1]))
    
    if option == 'face' or option == 'head':
        up_pos = [np.inf, -1]
        bottom_pos = [-np.inf, -1]
        left_pos = [-1, np.inf]
        right_pos = [-1, -np.inf]
        center = [0, 0]
        eye_pos = 0
        nose_pos = 0
        lip_pos = 0

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
                
                if key == 'left_eye' or key == 'right_eye':
                    eye_pos += i
                if key == 'nose_tip':
                    nose_pos += i
                if key == 'bottom_lip':
                    lip_pos += i
        
        eye_pos //= len(face_landmarks_list[0]['left_eye']) + len(face_landmarks_list[0]['right_eye'])
        nose_pos //= len(face_landmarks_list[0]['nose_tip'])
        lip_pos //= len(face_landmarks_list[0]['bottom_lip'])
        
        if option == 'head':
            center[0] = eye_pos
            center[1] = (left_pos[1] + right_pos[1]) // 2
            vertex = round(1.1 * (bottom_pos[0]) - center[0])
            covertex = round(1.5 * (right_pos[1] - center[1]))

            cv2.ellipse(mask, (center[1], center[0]), (covertex, vertex), 0, 0, 360, 1, -1)
            
            for t in range(10):
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
            center[0] = (eye_pos + nose_pos) // 2
            center[1] = (left_pos[1] + right_pos[1]) // 2
            vertex = round((0.5 * lip_pos + 0.5 * bottom_pos[0]) - center[0])
            covertex = right_pos[1] - center[1]
            cv2.ellipse(mask, (center[1], center[0]), (covertex, vertex), 0, 0, 360, 1, -1)
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
    img = cv2.imread('./tgt_2.jpg')
    mask = get_mask(img, 'face')
    cv2.imwrite("mask.jpg", mask)
