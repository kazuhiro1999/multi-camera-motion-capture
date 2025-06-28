import numpy as np
import scipy.stats as stats


KEYPOINTS_DICT = {
    'nose': 0,
    'left_inner_eye': 1,
    'left_eye': 2,
    'left_outer_eye': 3,
    'right_inner_eye': 4,
    'right_eye': 5,
    'right_outer_eye': 6,
    'left_ear': 7,
    'right_ear':8,
    'left_mouth': 9,
    'right_mouth': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_outer_hand': 17,
    'right_outer_hand': 18,
    'left_hand_tip': 19,
    'right_hand_tip': 20,
    'left_inner_hand': 21,
    'right_inner_hand': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_toe': 31,
    'right_toe': 32
}


def rotation_matrix_from_vectors(forward_vector, up_vector):
    
    if np.allclose(forward_vector, 0) or np.allclose(up_vector, 0):
        # Return the identity matrix if either vector is zero
        return np.identity(3)
        
    # Normalize the input vectors
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    up_vector = up_vector / np.linalg.norm(up_vector)
    
    # Compute the right vector using cross product
    right_vector = np.cross(forward_vector, up_vector)
    right_vector /= np.linalg.norm(right_vector)
    
    # Compute the actual up vector using cross product
    forward_vector = -np.cross(right_vector, up_vector)
    
    # Construct the rotation matrix
    R = np.column_stack((-right_vector, up_vector, forward_vector))
    
    return R
    

def matrix_to_quaternion(matrix):
    # Ensure the matrix is a numpy array
    matrix = np.asarray(matrix)
    
    # Calculate the trace of the matrix
    tr = matrix.trace()
    
    # Check the trace value to perform the suitable calculation
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (matrix[2, 1] - matrix[1, 2]) / S
        y = (matrix[0, 2] - matrix[2, 0]) / S
        z = (matrix[1, 0] - matrix[0, 1]) / S
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w = (matrix[2, 1] - matrix[1, 2]) / S
        x = 0.25 * S
        y = (matrix[0, 1] + matrix[1, 0]) / S
        z = (matrix[0, 2] + matrix[2, 0]) / S
    elif matrix[1, 1] > matrix[2, 2]:
        S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w = (matrix[0, 2] - matrix[2, 0]) / S
        x = (matrix[0, 1] + matrix[1, 0]) / S
        y = 0.25 * S
        z = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w = (matrix[1, 0] - matrix[0, 1]) / S
        x = (matrix[0, 2] + matrix[2, 0]) / S
        y = (matrix[1, 2] + matrix[2, 1]) / S
        z = 0.25 * S
        
    return x, y, z, w


def quaternion_to_matrix(quaternion):
    x, y, z, w = quaternion
    matrix = np.zeros([3,3])
    matrix[0,0] = 2*w**2 + 2*x**2 - 1
    matrix[0,1] = 2*x*y - 2*z*w
    matrix[0,2] = 2*x*z + 2*y*w
    matrix[1,0] = 2*x*y + 2*z*w
    matrix[1,1] = 2*w**2 + 2*y**2 - 1
    matrix[1,2] = 2*y*z - 2*x*w
    matrix[2,0] = 2*x*z - 2*y*w
    matrix[2,1] = 2*y*z + 2*x*w
    matrix[2,2] = 2*w**2 + 2*z**2 - 1
    return matrix
    

def extract_body_shape(keypoints3d_list):
    neck = (keypoints3d_list[:,KEYPOINTS_DICT['left_shoulder']] + keypoints3d_list[:,KEYPOINTS_DICT['right_shoulder']]) / 2
    hip = (keypoints3d_list[:,KEYPOINTS_DICT['left_hip']] + keypoints3d_list[:,KEYPOINTS_DICT['right_hip']]) / 2
    body_length = stats.norm.fit(np.linalg.norm(neck - hip, axis=-1))[0]
    
    head = stats.norm.fit(np.linalg.norm(keypoints3d_list[:,KEYPOINTS_DICT['nose']] - neck, axis=-1))[0] * 2
    
    l_upperarm_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['left_shoulder'], KEYPOINTS_DICT['left_elbow'])
    l_lowerarm_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['left_elbow'], KEYPOINTS_DICT['left_wrist'])
    r_upperarm_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['right_shoulder'], KEYPOINTS_DICT['right_elbow'])
    r_lowerarm_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['right_elbow'], KEYPOINTS_DICT['right_wrist'])
    
    l_upperleg_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['left_hip'], KEYPOINTS_DICT['left_knee'])
    l_lowerleg_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['left_knee'], KEYPOINTS_DICT['left_ankle'])
    r_upperleg_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['right_hip'], KEYPOINTS_DICT['right_knee'])
    r_lowerleg_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['right_knee'], KEYPOINTS_DICT['right_ankle'])
    
    arm_length = (l_upperarm_length + r_upperarm_length) / 2 + (l_lowerarm_length + r_lowerarm_length) / 2
    leg_length = (l_upperleg_length + r_upperleg_length) / 2 + (l_lowerleg_length + r_lowerleg_length) / 2
    height = leg_length + body_length + head
    return [height]


def calculate_bone_length(keypoints3d, joint_index_1, joint_index_2):
    distances = np.linalg.norm(keypoints3d[:, joint_index_1] - keypoints3d[:, joint_index_2], axis=-1)
    mu, std = stats.norm.fit(distances)
    return mu

