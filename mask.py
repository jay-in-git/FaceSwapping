import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_mask(img):
    face_landmarks_list = face_recognition.face_landmarks(img)
    mask = np.zeros((img.shape[0], img.shape[1]))
    
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

    center[0] = (eye_pos + nose_pos) // 2
    center[1] = (left_pos[1] + right_pos[1]) // 2
    vertex = round((0.5 * lip_pos + 0.5 * bottom_pos[0]) - center[0])
    covertex = right_pos[1] - center[1]

    cv2.ellipse(mask, (center[1], center[0]), (covertex, vertex), 0, 0, 360, 1, -1)
    return mask

if __name__ == '__main__':
    img = cv2.imread('./img/jay.png')
    mask = get_mask(img)
    cv2.imwrite("mask.jpg", mask)
