import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import torch
from copy import deepcopy

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch_3d(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)  # 转换为[1, B, 7]形状
    bb_test = np.expand_dims(bb_test, 1)  # 转换为[A, 1, 7]形状

    print('bb_test',bb_test.shape)
    print('bb_gt',bb_gt.shape)
    print('bb_test theta',bb_test[...,6])
    # 计算每个维度的重叠部分
    x1 = np.maximum(bb_test[..., 0] - bb_test[..., 3] / 2, bb_gt[..., 0] - bb_gt[..., 3] / 2)
    y1 = np.maximum(bb_test[..., 1] - bb_test[..., 4] / 2, bb_gt[..., 1] - bb_gt[..., 4] / 2)
    z1 = np.maximum(bb_test[..., 2] - bb_test[..., 5] / 2, bb_gt[..., 2] - bb_gt[..., 5] / 2)
    print('x1, y1, z1',x1.shape, y1.shape, z1.shape)
    x2 = np.minimum(bb_test[..., 0] + bb_test[..., 3] / 2, bb_gt[..., 0] + bb_gt[..., 3] / 2)
    y2 = np.minimum(bb_test[..., 1] + bb_test[..., 4] / 2, bb_gt[..., 1] + bb_gt[..., 4] / 2)
    z2 = np.minimum(bb_test[..., 2] + bb_test[..., 5] / 2, bb_gt[..., 2] + bb_gt[..., 5] / 2)

    # 计算每个维度的重叠长度
    w_overlap = np.maximum(0., x2 - x1)
    l_overlap = np.maximum(0., y2 - y1)
    h_overlap = np.maximum(0., z2 - z1)
    print('w_overlap', w_overlap.shape)

    # 计算重叠体积
    overlap_volume = w_overlap * l_overlap * h_overlap
    print('overlap_volume', overlap_volume.shape)
    print('overlap_volume', overlap_volume)
    # 计算每个边界框的体积
    bb_test_volume = bb_test[..., 3] * bb_test[..., 4] * bb_test[..., 5]
    bb_gt_volume = bb_gt[..., 3] * bb_gt[..., 4] * bb_gt[..., 5]

    # 计算联合体积
    union_volume = bb_test_volume + bb_gt_volume - overlap_volume

    # 计算IoU
    iou = overlap_volume / union_volume

    return iou

# def iou_batch_3d(bb_test, bb_gt):
#     """
#     计算简化版的3D IoU,考虑目标中心和尺寸,但不考虑旋转。
#     :param bb_test: 测试目标 [x, y, z, l, w, h, theta]
#     :param bb_gt: Ground truth目标 [x, y, z, l, w, h, theta]
#     :return: 简化版IoU
#     """
#     print('bb_test',bb_test.shape)
#     # 扩展为批量计算的维度 [B, 1, 7] 和 [1, B, 7]
#     bb_gt = np.expand_dims(bb_gt, 0)  # (1, B, 7)
#     bb_test = np.expand_dims(bb_test, 1)  # (A, 1, 7)

#     # 提取每个目标的中心坐标和尺寸 (x, y, z, l, w, h)
#     x1, y1, z1, l1, w1, h1, _ = np.split(bb_test, 7, axis=-1)
#     x2, y2, z2, l2, w2, h2, _ = np.split(bb_gt, 7, axis=-1)

#     print('x1, y1, z1',x1.shape, y1.shape, z1.shape)

#     # 计算每个目标的边界框范围
#     x1_min, x1_max = x1 - l1 / 2, x1 + l1 / 2
#     y1_min, y1_max = y1 - w1 / 2, y1 + w1 / 2
#     z1_min, z1_max = z1 - h1 / 2, z1 + h1 / 2
#     print('x1_min, y1_min, z1_min',x1_min.shape, y1_min.shape, z1_min.shape)
#     print('x1_max, y1_max, z1_max',x1_max.shape, y1_max.shape, z1_max.shape)

#     x2_min, x2_max = x2 - l2 / 2, x2 + l2 / 2
#     y2_min, y2_max = y2 - w2 / 2, y2 + w2 / 2
#     z2_min, z2_max = z2 - h2 / 2, z2 + h2 / 2

#     print('np.minimum(x1_max, x2_max)',np.minimum(x1_max, x2_max).shape)
#     print('np.minimum(x1_max, x2_max)- np.maximum(x1_min, x2_min)',(np.minimum(x1_max, x2_max) - np.maximum(x1_min, x2_min)).shape)
#     # 计算交集的宽、高、深度
#     inter_x = np.maximum(0, np.minimum(x1_max, x2_max) - np.maximum(x1_min, x2_min))
#     inter_y = np.maximum(0, np.minimum(y1_max, y2_max) - np.maximum(y1_min, y2_min))
#     inter_z = np.maximum(0, np.minimum(z1_max, z2_max) - np.maximum(z1_min, z2_min))
#     print('inter_x','inter_y','inter_z',inter_x.shape,inter_y.shape,inter_z.shape)

#     intersection = inter_x * inter_y * inter_z
#     print('intersection', intersection)

#     # 计算并集的体积
#     vol1 = l1 * w1 * h1
#     vol2 = l2 * w2 * h2
#     union = vol1 + vol2 - intersection

#     # 计算IoU
#     iou = np.squeeze(intersection / union)

    # return iou

class KalmanBoxTracker3D(object):
    count = 0

    def __init__(self, bbox, dt=0.1):
        self.dt = dt
        self.kf = KalmanFilter(dim_x=15, dim_z=7)  # Adjusted dimensions for 3D
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 0, 0, 0, self.dt, 0, 0, 0, 0.5*self.dt**2, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0, self.dt, 0, 0, 0, 0.5*self.dt**2, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0, 0, self.dt, 0, 0, 0, 0.5*self.dt**2, 0],  # z
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # l
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # w
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # h
            [0, 0, 0, 0, 0, 0 ,1, 0, 0, 0, self.dt, 0, 0, 0, 0.5*self.dt**2],  # theta
            [0, 0, 0, 0, 0, 0 ,0, 1, 0, 0, 0, self.dt, 0, 0, 0],  # x'
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, self.dt, 0, 0],  # y'
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, self.dt, 0],  # z'
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, self.dt],   # theta'
            [0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 1, 0, 0, 0],  # x''
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # y''
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # z''
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]   # theta''
        ])
        # Measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # z
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # l
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # w
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # h
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]   # theta
        ])

        # self.kf.R[3:, 3:] *= 10.  # Increase measurement uncertainty for volume and ratio
        # self.kf.P[7:, 7:] *= 1000.  # Increase initial state uncertainty for velocities and volume rate
        # self.kf.P *= 10.
        # self.kf.Q[7:, 7:] *= 0.01  # Process noise for the velocities and volume rate

        # Increase uncertainty for unobserved state variables
        self.kf.P = np.eye(15) * 10.0
        self.kf.P[7:11, 7:11] *= 100.0     # Velocity uncertainty
        self.kf.P[11:15, 11:15] *= 1000.0  # Acceleration uncertainty
        # Process noise
        self.kf.Q = np.eye(15) * 0.01
        self.kf.Q[7:11, 7:11] *= 0.1       # Velocity process noise
        self.kf.Q[11:15, 11:15] *= 1.0     # Acceleration process noise
        # Measurement noise
        self.kf.R = np.eye(7) * 1.0


        self.kf.x[:7] = self.convert_bbox_to_z(bbox)  # Initial state
        self.score = bbox[7]
        self.label = bbox[8]
        self.time_since_update = 0
        self.id = KalmanBoxTracker3D.count
        KalmanBoxTracker3D.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # New attribute to store the true trajectory (real bounding boxes)
        self.true_trajectory = []  # List to store real bounding boxes (tracked)
        self.true_trajectory.append(bbox)  # Add the initial true bbox
        self.kf_history = []  # 存储每个时刻的卡尔曼滤波器状态
        self.save_kf_state()  # 保存初始状态

    def save_kf_state(self):
        """保存当前卡尔曼滤波器的状态"""
        kf_state = {
            'x': self.kf.x.copy(),
            'P': self.kf.P.copy(),
            'timestamp': len(self.true_trajectory) - 1  # 对应的时间戳索引
        }
        self.kf_history.append(kf_state)

    def convert_bbox_to_z(self, bbox):
        # Convert a 3D bbox [cx, cy, cz, l, w, h] to state vector
        cx, cy, cz, l, w, h, theta, _, _ = bbox
        
        
        return np.array([cx, cy, cz, l, w, h, theta]).reshape((7, 1))

    def update(self, bbox):
        self.true_trajectory.append(bbox)  # Add real observed bbox to the trajectory
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = bbox[7]
        self.kf.update(self.convert_bbox_to_z(bbox))
        # 更新后保存状态
        self.save_kf_state()

    def predict(self):
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def predict_from_history(self, start_idx=0, steps=30, update_with_true=False):
        """
        从历史轨迹的指定时刻开始，使用当时的卡尔曼滤波器状态预测轨迹
        
        参数:
            start_idx: 历史轨迹的起始索引
            steps: 预测的时间步数
            update_with_true: 是否在预测时使用真实观测更新卡尔曼滤波器
            
        返回:
            预测的轨迹点列表
        """
        if start_idx >= len(self.kf_history):
            return []
        
        # 获取历史状态
        kf_state = self.kf_history[start_idx]
        
        # 创建卡尔曼滤波器的副本
        kf_copy = deepcopy(self.kf)
        
        # 设置为历史状态
        kf_copy.x = kf_state['x'].copy()
        kf_copy.P = kf_state['P'].copy()
        
        # 预测轨迹
        predicted_trajectory = []
        
        # 添加起始位置
        cx, cy, cz, l, w, h, theta = kf_copy.x[0:7].flatten()
        predicted_trajectory.append([cx, cy, cz, l, w, h, theta])
        
        # 确定真实轨迹的结束索引
        end_idx = min(start_idx + steps, len(self.true_trajectory))
        
        # 预测
        for i in range(start_idx + 1, end_idx):
            # 预测下一个状态
            kf_copy.predict()
            
            # 如果需要，使用真实观测更新
            if update_with_true and i < len(self.true_trajectory):
                z = self.convert_bbox_to_z(self.true_trajectory[i])
                kf_copy.update(z)
            
            # 从状态向量提取边界框
            cx, cy, cz, l, w, h, theta = kf_copy.x[0:7].flatten()
            
            # 添加到预测轨迹
            # print('predict trace:', cx, cy, cz, l, w, h, theta)
            predicted_trajectory.append([cx, cy, cz, l, w, h, theta])
        
        # 如果需要继续预测超过真实轨迹的长度
        for _ in range(end_idx, start_idx + steps):
            # 只预测，不更新
            kf_copy.predict()
            
            # 从状态向量提取边界框
            cx, cy, cz, l, w, h, theta = kf_copy.x[0:7].flatten()
            
            # 添加到预测轨迹
            predicted_trajectory.append([cx, cy, cz, l, w, h, theta])
        
        return predicted_trajectory
    
    def convert_x_to_bbox(self, x):
        # Convert state vector back to bbox
        cx, cy, cz, l, w, h, theta, _, _, _, _, _, _, _, _ = x.flatten()

        return np.array([cx, cy, cz, l, w, h, theta, self.score, self.label]).reshape((1, 9))

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)
    
    def get_true_trajectory(self):
        # Return the true trajectory (real bounding boxes)
        return self.true_trajectory
    
    def get_velocity(self):
        # Return velocity components from the state
        vx, vy, vz, vtheta = self.kf.x[7:11].flatten()
        return np.array([vx, vy, vz, vtheta])
    
    def get_acceleration(self):
        # Return acceleration components from the state
        ax, ay, az, atheta = self.kf.x[11:15].flatten()
        return np.array([ax, ay, az, atheta])
    
    def predict_future_position(self, time_steps):
        """
        Predict future position based on current state and acceleration model
        
        Args:
            time_steps: Number of time steps to predict into the future
            
        Returns:
            List of predicted positions [x, y, z, l, w, h, theta]
        """
        predictions = []
        current_state = np.copy(self.kf.x)
        dt = self.dt
        
        for t in range(time_steps):
            # Apply constant acceleration motion model
            x = current_state[0] + current_state[7]*dt + 0.5*current_state[11]*dt**2
            y = current_state[1] + current_state[8]*dt + 0.5*current_state[12]*dt**2
            z = current_state[2] + current_state[9]*dt + 0.5*current_state[13]*dt**2
            theta = current_state[6] + current_state[10]*dt + 0.5*current_state[14]*dt**2
            
            # Update velocities with acceleration
            vx = current_state[7] + current_state[11]*dt
            vy = current_state[8] + current_state[12]*dt
            vz = current_state[9] + current_state[13]*dt
            vtheta = current_state[10] + current_state[14]*dt
            
            # Keep size parameters constant
            l, w, h = current_state[3:6]
            
            # Create prediction
            pred_bbox = np.array([x, y, z, l, w, h, theta]).flatten()
            predictions.append(pred_bbox)
            
            # Update current state for next iteration
            current_state[0], current_state[1], current_state[2], current_state[6] = x, y, z, theta
            current_state[7], current_state[8], current_state[9], current_state[10] = vx, vy, vz, vtheta
            
        return predictions
    
def associate_detections_to_trackers_3d(detections, trackers, iou_threshold=0.3):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 7), dtype=int)

    # iou_matrix = iou_batch_3d(detections, trackers)
    device = torch.device("cuda")
    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(detections).float().to(device), torch.tensor(trackers).float().to(device)).cpu().numpy()
    # print('trackers', trackers.shape)()
    # print('iou_matrix shape', iou_matrix.shape)
    # print('iou_matrix', iou_matrix)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class SORT3D(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 9))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            # print('pos',pos.shape)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers_3d(dets[:,:7], trks, self.iou_threshold)
        # print('matched', 'unmatched_dets', matched, unmatched_dets)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker3D(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # print('d', d.shape)
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate(([trk.id], d)).reshape(1, -1))
                # Include the tracker's history in the output
                # print('trk.id',trk.id)
                # history_array = np.array(trk.get_true_trajectory())
                # print('history_array',history_array.shape)
                # ret.append(np.concatenate(([trk.id + 1], d, history_array.flatten())).reshape(1, -1))
                
           
            i -= 1
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 8))

if __name__ == '__main__':
    rospy.init_node('sort_3d_tracker', anonymous=True)
    marker_pub = rospy.Publisher('tracked_objects', Marker, queue_size=10)

    tracker = SORT3D()

    # 假设检测框数据为 (x, y, z, w, h, d)
    detections = [
        (1, 2, 3, 0.5, 0.5, 0.5),
        (4, 5, 6, 0.5, 0.5, 0.5),
    ]

    rospy.spin()
    tracked_objects = tracker.update(detections)
    # publish_markers(tracked_objects, marker_pub)

