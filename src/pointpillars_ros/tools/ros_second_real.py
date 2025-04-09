#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray
import ros_numpy

import time
import numpy as np
from pyquaternion import Quaternion

import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch
import scipy.linalg as linalg

import sys
sys.path.append("/home/pointpillars_ros/src/pointpillars_ros")

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from sort import SORT3D
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_matrix

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path: 根目录
            dataset_cfg: 数据集配置
            class_names: 类别名称
            training: 训练模式
            logger: 日志
            ext: 扩展名
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        

class Pointpillars_ROS:
    def __init__(self):
        rospy.init_node('pointpillars_ros_node', anonymous=True)
        self.frame_count = 0
        self.fps = 0
        self.total_frames = 0
        # self.prev_bboxes= {}
        # self.prev_time = None
        self.total_preprocess_time = 0
        self.total_inference_time = 0
        self.total_time_elapsed = 0
        self.prediction_horizon = 30  # 预测未来多少帧
        config_path, ckpt_path = self.init_ros()
        self.init_pointpillars(config_path, ckpt_path)
        # 添加BEV相关参数
        self.x_range = [-50, 50]  # BEV平面x范围(m)
        self.y_range = [-50, 50]  # BEV平面y范围(m)
        self.resolution = 0.1  # 网格分辨率(m/像素)
        
        # 计算网格尺寸
        self.x_size = int((self.x_range[1] - self.x_range[0]) / self.resolution)
        self.y_size = int((self.y_range[1] - self.y_range[0]) / self.resolution)

        self.sort3d_tracker = SORT3D(max_age=3, min_hits=3)
        self.last_time = time.time()



        rospy.on_shutdown(self.on_shutdown)

    def on_shutdown(self):
        self.tf_buffer.clear()
        rospy.loginfo("Shutting down, 清除缓存")

    def init_ros(self):
        """ Initialize ros parameters """
        config_path = rospy.get_param("/config_path", "/home/pointpillars_ros/src/pointpillars_ros/tools/cfgs/kitti_models/PartA2_free.yaml")
        ckpt_path = rospy.get_param("/ckpt_path", "/home/pointpillars_ros/src/pointpillars_ros/tools/OpenPCDetZoo/PartA2_free_7872.pth")
        self.sub_velo = rospy.Subscriber("/rslidar_points", PointCloud2, self.lidar_callback, queue_size=2,  buff_size=2**12)
        self.pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=2)
        # self.pub_marker = rospy.Publisher('/visualization_Marker', MarkerArray, queue_size=1)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=2)
        self.marker_history_pub = rospy.Publisher('/visualization_history_array', MarkerArray, queue_size=2)
        self.marker_prediction_pub = rospy.Publisher('/visualization_prediction_array', MarkerArray, queue_size=2)
        # 添加BEV相关发布器
        self.pub_bev = rospy.Publisher('/bev_grid', OccupancyGrid, queue_size=1)
        self.pub_bev_boxes = rospy.Publisher('/bev_boxes', MarkerArray, queue_size=1)
        # Create a publisher for world frame detections
        self.pub_world_bbox = rospy.Publisher("/world_detections", BoundingBoxArray, queue_size=2)
        # Add TF buffer and listener for coordinate transformation
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        print(ckpt_path)
        return config_path, ckpt_path


    def init_pointpillars(self, config_path, ckpt_path):
        """ Initialize second model """
        logger = common_utils.create_logger() # 创建日志
        logger.info('-----------------Quick Demo of Pointpillars-------------------------')
        cfg_from_yaml_file(config_path, cfg)  # 加载配置文件
        
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            ext='.bin', logger=logger
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        # 加载权重文件
        self.model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
        logger.info('Model loaded successfully.')
    
        # 等待一段时间，确保模型完全加载
        wait_time = 5  # 设置等待时间，例如 5 秒
        logger.info(f"Waiting for {wait_time} seconds to ensure the model is fully loaded...")
        time.sleep(wait_time)  # 等待一段时间
        self.model.cuda() # 将网络放到GPU上
        self.model.eval() # 开启评估模式

    # def rotate_mat(self, axis, radian):
    #     rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    #     return rot_matrix
    # 修改处理轨迹的部分

    # def transform_points_to_world(self, points, msg_stamp):
    #     """
    #     将点从velo_link坐标系转换到world坐标系
        
    #     参数:
    #     points: numpy数组，形状为(N, 3)，表示N个点的[x, y, z]坐标
    #     msg_stamp: 消息时间戳
        
    #     返回:
    #     world_points: numpy数组，形状为(N, 3)，表示转换后的点
    #     """
    #     try:
    #         # 获取从velo_link到world的变换
    #         transform = self.tf_buffer.lookup_transform(
    #             'world',              # 目标坐标系
    #             'velo_link',          # 源坐标系
    #             rospy.Time(0),            # 时间戳
    #             rospy.Duration(0.1)   # 超时时间
    #         )
            
    #         # 提取变换参数
    #         trans = transform.transform.translation
    #         rot = transform.transform.rotation
            
    #         # 创建变换矩阵
    #         from tf.transformations import quaternion_matrix
    #         quat = [rot.x, rot.y, rot.z, rot.w]
    #         mat = quaternion_matrix(quat)
    #         mat[0:3, 3] = [trans.x, trans.y, trans.z]
            
    #         # 转换点
    #         world_points = np.zeros_like(points)
    #         for i in range(len(points)):
    #             # 齐次坐标
    #             point_h = np.append(points[i], 1.0)
    #             # 变换
    #             world_point_h = np.dot(mat, point_h)
    #             # 取前3个元素
    #             world_points[i] = world_point_h[:3]
            
    #         return world_points
            
    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
    #         rospy.logwarn(f"TF转换失败: {e}")
    #         return points  
    
    def transform_3d_boxes_to_world(self, boxes, time_stamp):
        """
        批量将3D检测框从雷达坐标系转换到世界坐标系
        
        参数:
        boxes: 形状为[N,7]的数组，每行为[x,y,z,l,w,h,heading]格式的3D检测框
        time_stamp: 时间戳，用于查询变换
        
        返回:
        world_boxes: 形状为[N,7]的数组，世界坐标系下的3D检测框
        """
        # try:
            # 从tf获取雷达坐标系到世界坐标系的变换
        transform = self.tf_buffer.lookup_transform(
            'map',              # 目标坐标系
            'rslidar',          # 源坐标系
            rospy.Time(0),           # 时间戳
            rospy.Duration(0.1)   # 超时时间
        )
        
        # 从transform中提取平移和旋转信息
        translation = [transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z]
        
        rotation = [transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w]
        
        # 构建变换矩阵
        tf_matrix = quaternion_matrix(rotation)
        tf_matrix[0, 3] = translation[0]
        tf_matrix[1, 3] = translation[1]
        tf_matrix[2, 3] = translation[2]
        
        # 提取旋转矩阵部分
        rotation_matrix = tf_matrix[:3, :3]
        
        # 转换为四元数表示
        q_transform = Quaternion(matrix=rotation_matrix)
        
        # 初始化结果数组
        world_boxes = np.zeros_like(boxes)
        
        # 批量处理中心点坐标
        positions = np.hstack([boxes[:, :3], np.ones((boxes.shape[0], 1))])  # [N,4]，齐次坐标
        world_positions = np.dot(positions, tf_matrix.T)  # [N,4]
        world_boxes[:, :3] = world_positions[:, :3]  # 更新位置
        
        # 保持尺寸不变
        world_boxes[:, 3:6] = boxes[:, 3:6]
        
        # 逐个处理heading（因为四元数操作需要单独处理）
        for i in range(boxes.shape[0]):
            heading = boxes[i, 6]
            q_lidar = Quaternion(axis=(0, 0, 1), radians=float(heading))
            q_world = q_transform * q_lidar
            world_boxes[i, 6] = q_world.radians
        
        return world_boxes
            
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        #     rospy.logwarn(f"无法获取坐标变换: {e}")
        #     return boxes
        
    def lidar_callback(self, msg):
        """ Captures pointcloud data and feed into second model for inference """
        call_time = time.time()
        # pcl_msg = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z","intensity","ring"))
        # np_p = np.array(list(pcl_msg), dtype=np.float32)
        np_p = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        # np_p = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        # np_p = np_p[['x','y','z']].view(np.float32)
        print('np shape',np_p.shape)
        # # 旋转轴
        # rand_axis = [0,1,0]
        # #旋转角度0.1047
        # yaw = 0
        # #返回旋转矩阵
        # rot_matrix = self.rotate_mat(rand_axis, yaw)
        # np_p_rot = np.dot(rot_matrix, np_p[:,:3].T).T
        
        # convert to xyzi point cloud
        # x = np_p['x'].reshape(-1)
        # y = np_p['y'].reshape(-1)
        # z = np_p['z'].reshape(-1)
        i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        x = np_p[:, 0].reshape(-1)
        y = np_p[:, 1].reshape(-1)
        z = np_p[:, 2].reshape(-1)
        # if np_p.shape[1] == 4: # if intensity field exists
        #     i = np_p[:, 3].reshape(-1)
        # else:
        #     i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        points = np.stack((x, y, z, i)).T
        print(points.shape)
        # 组装数组字典
        input_dict = {
            'points': points,
            'frame_id': 0,
        }
        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict) # 数据预处理
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict) # 将数据放到GPU上
        preprocess_end_time = time.time() 
        preprocess_time = preprocess_end_time - call_time
        self.total_preprocess_time += preprocess_time  # 累加数据处理时间

        
        #inference time
        start_time = time.time()
        pred_dicts, _ = self.model.forward(data_dict) # 模型前向传播
        end_time = time.time()
        inference_time = end_time-start_time
        self.total_inference_time += inference_time  # 累加推理时间
        self.frame_count += 1

        #calculate FPS
        call_end_time = time.time()
        time_elapsed = call_end_time - call_time
    
        self.total_time_elapsed += time_elapsed  # 累计总时间
        self.total_frames += 1 # 累加 FPS
        # 每1秒输出一次平均数据
        if self.total_frames > 0 and call_end_time - self.last_time >= 1.0:
            avg_preprocess_time = self.total_preprocess_time / self.total_frames
            avg_inference_time = self.total_inference_time / self.total_frames
            avg_fps = self.total_frames / self.total_time_elapsed 

            rospy.loginfo("Average Processing Time: %.4f seconds, Average Inference Time: %.4f seconds, Average FPS: %.2f", avg_preprocess_time, avg_inference_time, avg_fps)

            self.last_time = call_end_time
            # 重置计数器
            self.total_preprocess_time = 0.0
            self.total_inference_time = 0.0
            self.total_time_elapsed  = 0.0
            self.total_frames = 0

        scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
        mask = scores>0.5
        scores = scores[mask]
        boxes_lidar = pred_dicts[0]['pred_boxes'][mask].detach().cpu().numpy()
        label = pred_dicts[0]['pred_labels'][mask].detach().cpu().numpy()
        num_detections = boxes_lidar.shape[0]
        # rospy.loginfo("The num is: %d ", num_detections)

        # print(boxes_lidar)
        # print(scores)
        # print('label',label)

        # calculate speed
        # current_time = time.time()
        # _time = current_time - self.prev_time if self.prev_time else 0.1



        # 投影点云到BEV平面
        bev_grid = self.project_points_to_bev(np_p)
        self.publish_bev_grid(bev_grid, msg.header)
        
        # 投影检测框到BEV平面
        if num_detections > 0:
            bev_corners = self.project_boxes_to_bev(boxes_lidar)
            self.publish_bev_boxes(bev_corners, msg.header)

        # 更新SORT3D跟踪器
        # print('box lidar',boxes_lidar.shape)
        world_boxes_lidar = self.transform_3d_boxes_to_world(boxes_lidar, msg.header.stamp)
        # print('box lidar',boxes_lidar.shape)
        # print('world box lidar',world_boxes_lidar.shape)
        tracked_objects = self.sort3d_tracker.update(world_boxes_lidar)


        # tracked_objects_trace = self.sort3d_tracker.trackers
        if len(tracked_objects)>0 :
            # print('tracked objects', tracked_objects.shape, len(self.sort3d_tracker.trackers))
            # Visualize the results in RViz
            self.visualize(tracked_objects, msg=msg)
            history_list = []
            predict_list = []
            for tracker in tracked_objects:
                id = tracker[0]
                for trk in self.sort3d_tracker.trackers:
                    if id == trk.id+1:
                        history_array = np.array(trk.get_true_trajectory())
                        # if history_array.size > 0:
                        #     world_history_array = self.transform_points_to_world(
                        #         history_array[:,:3], msg.header.stamp)
                        #     print('world array',world_history_array)
                        #     print('history_array',history_array[:,:3])
                        #     history_list.append(world_history_array)

                        # predict_array = np.array(trk.predict_future_position(self.prediction_horizon))
                        # if predict_array.size > 0:
                        #     world_predict_array = self.transform_points_to_world(
                        #         predict_array, msg.header.stamp)
                        #     predict_list.append(world_predict_array)
                        # print('history_array out',history_array.shape)
                        predict_array = trk.predict_from_history(start_idx=len(trk.get_true_trajectory())//2, steps=30, update_with_true=False)
                        history_list.append(history_array)
                        predict_list.append(predict_array)
            
            self.visualize_history_trace(history_list, msg=msg)
            self.visualize_prediction(predict_list, msg=msg)

        # box_time = time.time()
        arr_bbox = BoundingBoxArray()
        # marker_arr = MarkerArray()
        for i in range(num_detections):
            bbox = BoundingBox()

            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = msg.header.stamp
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2]) + float(boxes_lidar[i][5]) / 2
            bbox.dimensions.x = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z = float(boxes_lidar[i][5])  # height
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = label[i]

            arr_bbox.boxes.append(bbox)
        
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = msg.header.stamp


        # update time
        # self.prev_time = current_time
        # rospy.loginfo("boundingbox time: %.4f", time.time()-call_time)
        
        self.pub_bbox.publish(arr_bbox)
        
        # Now transform and publish in world coordinates
        try:
            # Get the transform from velodyne frame to world frame
            transform = self.tf_buffer.lookup_transform('map', msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            
            # Create a new BoundingBoxArray for world coordinates
            world_arr_bbox = BoundingBoxArray()
            world_arr_bbox.header.frame_id = 'map'
            world_arr_bbox.header.stamp = msg.header.stamp
            
            for bbox in arr_bbox.boxes:
                # Create a PoseStamped for the bounding box
                bbox_pose = PoseStamped()
                bbox_pose.header = bbox.header
                bbox_pose.pose = bbox.pose
                
                # Transform the pose to world coordinates
                transformed_pose = tf2_geometry_msgs.do_transform_pose(bbox_pose, transform)
                
                # Create a new bounding box with transformed pose
                world_bbox = BoundingBox()
                world_bbox.header.frame_id = 'map'
                world_bbox.header.stamp = msg.header.stamp
                world_bbox.pose = transformed_pose.pose
                world_bbox.dimensions = bbox.dimensions  # Keep the same dimensions
                world_bbox.value = bbox.value  # Keep the same confidence score
                world_bbox.label = bbox.label  # Keep the same label
                
                world_arr_bbox.boxes.append(world_bbox)
            
            # Publish the transformed bounding boxes
            self.pub_world_bbox.publish(world_arr_bbox)
            # rospy.loginfo("Published %d bounding boxes in world frame", len(world_arr_bbox.boxes))
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Failed to transform bounding boxes to world frame: %s", str(e))

    def clear_markers(self):
        # 创建一个MarkerArray消息，用于删除所有旧的标记
        clear_marker = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        clear_marker.markers.append(marker)
        self.marker_pub.publish(clear_marker)
        self.marker_history_pub.publish(clear_marker)
        self.marker_prediction_pub.publish(clear_marker)
        self.pub_bev_boxes .publish(clear_marker)

    def visualize(self, tracked_objects, msg):
        self.clear_markers()
        marker_array = MarkerArray()
        for obj in tracked_objects:
            # 创建立方体标记
            cube_marker = Marker()
            cube_marker.header.frame_id = msg.header.frame_id
            cube_marker.header.stamp = msg.header.stamp
            cube_marker.ns = "tracked_objects"
            cube_marker.id = int(obj[0])
            cube_marker.type = Marker.CUBE
            cube_marker.action = Marker.ADD
            cube_marker.pose.position = Point(obj[1], obj[2], obj[3])
            q = Quaternion(axis=(0, 0, 1), radians=float(obj[7]))
            cube_marker.pose.orientation.x = q.x
            cube_marker.pose.orientation.y = q.y
            cube_marker.pose.orientation.z = q.z
            cube_marker.pose.orientation.w = q.w
            cube_marker.scale.x = obj[4]
            cube_marker.scale.y = obj[5]
            cube_marker.scale.z = obj[6]
            cube_marker.color = ColorRGBA(0, 1, 0, 1)  # Green color for tracked objects
            marker_array.markers.append(cube_marker)

            # 创建文本标记
            text_marker = Marker()
            text_marker.header.frame_id = msg.header.frame_id
            text_marker.header.stamp = msg.header.stamp
            text_marker.ns = "tracked_objects_text"
            text_marker.id = int(obj[0]) + 1000  # 确保文本标记的 ID 不与立方体标记冲突
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = Point(obj[1]+obj[4]/2, obj[2]+obj[5]/2, obj[3] + obj[6]/2)  # 将文本稍微提高以避免与立方体重叠
            text_marker.scale.z = 2  # 设置文本大小
            text_marker.color = ColorRGBA(1, 1, 1, 1)  # White color for text
            text_marker.text = f"ID: {int(obj[0])}"
            marker_array.markers.append(text_marker)

            # 发布文本标记
        self.marker_pub.publish(marker_array)

    def visualize_history_trace(self, history_array_list, msg):
        self.clear_markers()
        marker_array = MarkerArray()
        for i,history_array in enumerate(history_array_list):
            # 创建一个线条标记
            line_marker = Marker()
            line_marker.header.frame_id = "map"
            line_marker.header.stamp = msg.header.stamp
            line_marker.ns = "history_objects"
            line_marker.id = i  
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.1  # 线宽
            line_marker.color = ColorRGBA(1, 0, 0, 1)  # 红色
            

            # 添加所有点到 line_marker
            for obj in history_array:
                point = Point(obj[0], obj[1], obj[2])  
                line_marker.points.append(point)

            # 将线条标记添加到 MarkerArray
            marker_array.markers.append(line_marker)

        # 发布 MarkerArray
        self.marker_history_pub.publish(marker_array)

    def visualize_prediction(self, predict_array_list, msg):
        self.clear_markers()
        marker_array = MarkerArray()
        
        # 创建箭头标记 (用于显示速度和方向)
        for j, predict_array in enumerate(predict_array_list):
            for i, pred_pos in enumerate(predict_array):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = msg.header.stamp
                marker.ns = f"predicted_trajectory_{j}"
                marker.id = j*1000+i
                
                if i < len(predict_array)-1:
                    # 创建线段连接相邻预测点
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    
                    # 添加当前点和下一个点
                    p1 = Point()
                    p1.x = pred_pos[0]
                    p1.y = pred_pos[1]
                    p1.z = pred_pos[2]
                    
                    p2 = Point()
                    p2.x = predict_array[i+1][0]
                    p2.y = predict_array[i+1][1]
                    p2.z = predict_array[i+1][2]
                    
                    marker.points = [p1, p2]
                    
                    # 设置线宽
                    marker.scale.x = 0.1  # 线宽
                    
                    # 使用渐变颜色 (从绿色到红色)
                    alpha = float(i) / (len(predict_array) - 1)
                    marker.color.r = alpha
                    marker.color.g = 1.0 - alpha
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                else:
                    # 为最后一个预测点创建小立方体
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    # print('pred_pos',pred_pos)
                    x, y, z, l, w, h, theta = pred_pos
                    
                    marker.pose.position.x = x
                    marker.pose.position.y = y
                    marker.pose.position.z = z
                    
                    q = Quaternion(axis=(0, 0, 1), radians=float(theta))
                    marker.pose.orientation.x = q[0]
                    marker.pose.orientation.y = q[1]
                    marker.pose.orientation.z = q[2]
                    marker.pose.orientation.w = q[3]
                    
                    marker.scale.x = l
                    marker.scale.y = w
                    marker.scale.z = h
                    
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.6
                marker_array.markers.append(marker)    
        # 发布标记数组
        self.marker_prediction_pub.publish(marker_array)

    def project_points_to_bev(self, points):
        """
        将点云投影到BEV平面
        points: [N, 3+] 点云数组
        返回: 2D栅格地图表示BEV视图
        """
        # 创建空的BEV栅格图
        bev_map = np.zeros((self.x_size, self.y_size), dtype=np.int8)
        
        # 过滤点云(只保留感兴趣区域内的点)
        mask_x = (points[:, 0] >= self.x_range[0]) & (points[:, 0] < self.x_range[1])
        mask_y = (points[:, 1] >= self.y_range[0]) & (points[:, 1] < self.y_range[1])
        mask = mask_x & mask_y
        valid_points = points[mask]
        
        if len(valid_points) == 0:
            return bev_map
        
        # 转换点云坐标到BEV网格索引
        x_indices = ((valid_points[:, 0] - self.x_range[0]) / self.resolution).astype(np.int32)
        y_indices = ((valid_points[:, 1] - self.y_range[0]) / self.resolution).astype(np.int32)
        
        # 限制索引范围
        x_indices = np.clip(x_indices, 0, self.x_size - 1)
        y_indices = np.clip(y_indices, 0, self.y_size - 1)
        
        # 填充BEV图(可以根据高度或密度编码更多信息)
        # 这里使用简单的计数方式，将点云密度映射到栅格值
        for i in range(len(valid_points)):
            bev_map[y_indices[i], x_indices[i]] = 100  # 设置为已占据(100表示占据)
            
        return bev_map
    
    def publish_bev_grid(self, bev_map, header):
        """发布BEV栅格地图用于RViz显示"""
        grid = OccupancyGrid()
        grid.header = header
        # grid.header.frame_id = header.frame_id
        grid.info.resolution = self.resolution
        grid.info.width = self.x_size
        grid.info.height = self.y_size
        grid.info.origin.position.x = self.x_range[0]
        grid.info.origin.position.y = self.y_range[0]
        grid.info.origin.position.z = 0.0
        
        # 填充栅格数据
        grid.data = bev_map.flatten().tolist()
        
        # 发布消息
        self.pub_bev.publish(grid)
    
    def project_boxes_to_bev(self, boxes_lidar):
        """
        将3D检测框投影到BEV平面
        boxes_lidar: [N, 7] 每行为 [x, y, z, l, w, h, heading]
        返回: 2D框的顶点坐标 [N, 4, 2]
        """
        num_boxes = boxes_lidar.shape[0]
        corners = np.zeros((num_boxes, 4, 2))
        
        for i in range(num_boxes):
            # 提取3D框参数 - 保持与您原始代码中的顺序一致
            x, y, z = boxes_lidar[i][0], boxes_lidar[i][1], boxes_lidar[i][2]  # 中心坐标
            l, w, h = boxes_lidar[i][3], boxes_lidar[i][4], boxes_lidar[i][5]  # 长宽高
            heading = boxes_lidar[i][6]  # 朝向角
            
            # 计算未旋转的框的四个角点(相对于中心)
            # 注意：根据您的代码，l对应x维度(前后)，w对应y维度(左右)
            box_corners = np.array([
                [l/2, w/2],   # 右前
                [l/2, -w/2],  # 右后
                [-l/2, -w/2], # 左后
                [-l/2, w/2]   # 左前
            ])
            
            # 应用顺时针旋转
            cos_rot, sin_rot = np.cos(heading), np.sin(heading)
            rot_matrix = np.array([
                [cos_rot, sin_rot],
                [-sin_rot, cos_rot]
            ])
            
            # 旋转并平移角点
            for j in range(4):
                corners[i, j] = np.dot(box_corners[j], rot_matrix) + np.array([x, y])
                    
        return corners

    def publish_bev_boxes(self, corners, header):
        """
        将BEV投影的检测框发布为MarkerArray用于RViz可视化
        corners: [N, 4, 2] 每个框的四个角点
        """
        self.clear_markers()
        marker_array = MarkerArray()
        
        for i, box_corners in enumerate(corners):
            # 创建线条Marker
            marker = Marker()
            marker.header = header
            # marker.header.frame_id = header.frame_id  # 使用与点云相同的坐标系
            marker.ns = "bev_boxes"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1  # 线宽
            
            # 设置颜色(可以根据类别或置信度设置不同颜色)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            
            # 添加角点(回到第一个点形成闭环)
            for corner in box_corners:
                p = Point()
                p.x = float(corner[0])
                p.y = float(corner[1])
                p.z = 0.1  # 稍微高于地面，便于可视化
                marker.points.append(p)
            
            # 添加第一个点以闭合线条
            p = Point()
            p.x = float(box_corners[0][0])
            p.y = float(box_corners[0][1])
            p.z = 0.1
            marker.points.append(p)
        
            marker_array.markers.append(marker)
    
        self.pub_bev_boxes.publish(marker_array)

if __name__ == '__main__':
    # rospy.init_node('pointpillars_ros_node', anonymous=True)
    try:
        sec = Pointpillars_ROS()
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
