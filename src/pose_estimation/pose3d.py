import numpy as np
#import tensorflow as tf


def recover_pose_3d(proj_matrices, keypoints2d_list, th=0.5):
    """
    Recover 3D poses from 2D keypoints.
    
    Parameters
    ----------
    proj_matrices : array
        Projection matrices of the cameras.
    keypoints2d_list : array
        2D coordinates of each keypoint from each camera view.
    th : float, optional
        Confidence threshold. Only keypoints with confidence above this threshold are considered.
        
    Returns
    -------
    keypoints3d : array
        3D coordinates of each keypoint.
    failed_joints : list
        list of joint that failed to triangulate with SVD.
    """

    if len(keypoints2d_list) < 2: # At least two viewpoints are required for 3D reconstruction
        return None
    proj_matrices = np.array(proj_matrices)
    keypoints2d_list = np.array(keypoints2d_list)
    n_views, n_joints, _ = keypoints2d_list.shape
    if proj_matrices.shape[0] != n_views:
        return None  # Number of projection matrices does not match number of views.

    keypoints3d = []
    failed_joints = []
    for joint_i in range(n_joints):
        points = keypoints2d_list[:,joint_i,:2]
        confidences = keypoints2d_list[:,joint_i,2]
        confidences = np.where(confidences < 0.5, 0, confidences)
        alg_confidences = confidences / (confidences.sum() + 1e-5)
        ret, point_3d = triangulate_points_np(proj_matrices, points, alg_confidences)
        keypoints3d.append(point_3d)

        if not ret:
            failed_joints.append(joint_i)

    return np.array(keypoints3d), failed_joints
    
# SVD法
def triangulate_points_np(proj_matrices, points, confidences):
    n_views = len(proj_matrices)
    
    # 行列Aを構築
    A = np.tile(proj_matrices[:, 2:3], (1, 2, 1)) * points[:, :, None]
    A -= proj_matrices[:, :2]
    A *= confidences[:, None, None]

    # AをリシェイプしてSVDを計算
    A_reshaped = A.reshape(-1, 4)
    # SVD計算を例外処理で保護
    try:
        u, s, vh = np.linalg.svd(A_reshaped, full_matrices=False)
        # vhの最終列を取得し、3次元点を計算
        point_3d_homo = -vh[-1, :]  # 最終列を取得
        point_3d = point_3d_homo[:-1] / point_3d_homo[-1]  # 同次座標をデカルト座標に変換
    except np.linalg.LinAlgError:
        # SVD失敗時はゼロに設定
        return False, np.zeros(3)

    return True, point_3d

"""def triangulate_points_tf(proj_matrices, points, confidences):
    n_views = len(proj_matrices)
    
    A = tf.cast(tf.tile(proj_matrices[:, 2:3], (1,2,1)), dtype=tf.float32) * tf.reshape(tf.cast(points, dtype=tf.float32), [n_views, 2, 1])
    A -= tf.cast(proj_matrices[:, :2], dtype=tf.float32)
    A *= tf.reshape(tf.cast(confidences, dtype=tf.float32), [-1, 1, 1])

    u, s, v = tf.linalg.svd(tf.reshape(A, [-1, 4]), full_matrices=False)
    vh = tf.linalg.adjoint(v)

    point_3d_homo = -tf.transpose(vh)[None,:,3]   
    point_3d = tf.transpose(tf.transpose(point_3d_homo)[:-1] / tf.transpose(point_3d_homo)[-1])[0]
    return point_3d"""

# 再投影
def reprojection(keypoints3d, proj_matrix):
    # keypoints3d : (num_keypoints, 3)
    assert keypoints3d.shape[1] == 3
    num_keypoints = keypoints3d.shape[0]
    points3d = np.vstack((keypoints3d.T, np.ones((1,num_keypoints))))
    keypoints2d = proj_matrix @ points3d
    keypoints2d = keypoints2d[:2,:] / keypoints2d[2,:]
    return keypoints2d.T

# カメラの射影行列を計算
def calculate_projection_matrix(intrinsic, extrinsic):
    """Calculate projection matrix from intrinsic and extrinsic parameters."""
    return np.dot(intrinsic, extrinsic)