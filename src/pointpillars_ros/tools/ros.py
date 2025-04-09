#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
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
        config_path, ckpt_path = self.init_ros()
        self.init_pointpillars(config_path, ckpt_path)
        self.sort3d_tracker = SORT3D(max_age=3, min_hits=3)
        self.last_time = time.time()
        # print("init ros")
        self.frame_count = 0
        self.fps = 0
        self.total_frames = 0
        # self.prev_bboxes= {}
        # self.prev_time = None
        self.total_preprocess_time = 0
        self.total_inference_time = 0
        self.total_time_elapsed = 0


    def init_ros(self):
        """ Initialize ros parameters """
        config_path = rospy.get_param("/config_path", "/home/pointpillars_ros/src/pointpillars_ros/tools/cfgs/kitti_models/pointpillar.yaml")
        ckpt_path = rospy.get_param("/ckpt_path", "/home/pointpillars_ros/src/pointpillars_ros/tools/OpenPCDetZoo/pointpillar_7728.pth")
        self.sub_velo = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, self.lidar_callback, queue_size=2,  buff_size=2**12)
        self.pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=2)
        # self.pub_marker = rospy.Publisher('/visualization_Marker', MarkerArray, queue_size=1)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=2)
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
        self.model.cuda() # 将网络放到GPU上
        self.model.eval() # 开启评估模式


    def rotate_mat(self, axis, radian):
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        return rot_matrix

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
        rospy.loginfo("The num is: %d ", num_detections)

        # print(boxes_lidar)
        # print(scores)
        print('label',label)

        # calculate speed
        # current_time = time.time()
        # _time = current_time - self.prev_time if self.prev_time else 0.1

        # 更新SORT3D跟踪器
        # print('box lidar',boxes_lidar.shape)
        tracked_objects = self.sort3d_tracker.update(boxes_lidar)
        if len(tracked_objects)>0 :
            print('tracked objects', tracked_objects.shape)
            # Visualize the results in RViz
            self.visualize(tracked_objects)

        box_time = time.time()
        arr_bbox = BoundingBoxArray()
        # marker_arr = MarkerArray()
        for i in range(num_detections):
            bbox = BoundingBox()

            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
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
        arr_bbox.header.stamp = rospy.Time.now()

        # update time
        # self.prev_time = current_time
        rospy.loginfo("boundingbox time: %.4f", time.time()-call_time)
        
        self.pub_bbox.publish(arr_bbox)
        
        # self.pub_marker.publish(marker_arr)

    def clear_markers(self):
        # 创建一个MarkerArray消息，用于删除所有旧的标记
        clear_marker = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        clear_marker.markers.append(marker)
        self.marker_pub.publish(clear_marker)

    def visualize(self, tracked_objects):
        self.clear_markers()
        marker_array = MarkerArray()
        for obj in tracked_objects:
            # 创建立方体标记
            cube_marker = Marker()
            cube_marker.header.frame_id = "velo_link"
            cube_marker.header.stamp = rospy.Time.now()
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
            text_marker.header.frame_id = "velo_link"
            text_marker.header.stamp = rospy.Time.now()
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


if __name__ == '__main__':
    sec = Pointpillars_ROS()
    rospy.init_node('pointpillars_ros_node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
