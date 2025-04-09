#!/usr/bin/env python
import torch
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid

class PointCloudToBEV:
    def __init__(self):
        self.x_range = [-50, 50]  # BEV平面x范围(m)
        self.y_range = [-50, 50]  # BEV平面y范围(m)
        self.resolution = 0.1  # 网格分辨率(m/像素)
        self.height_threshold = [-2, 4]  # 高度过滤阈值
        
        # 计算网格尺寸
        self.x_size = int((self.x_range[1] - self.x_range[0]) / self.resolution)
        self.y_size = int((self.y_range[1] - self.y_range[0]) / self.resolution)
        
        # 创建ROS订阅者和发布者
        self.sub_pc = rospy.Subscriber('/kitti/velo/pointcloud', PointCloud2, self.callback)
        self.pub_bev = rospy.Publisher('/bev_grid', OccupancyGrid, queue_size=1)
        
    def callback(self, msg):
        # 将ROS点云转换为numpy数组
        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        
        # 转换为PyTorch张量
        points_tensor = torch.from_numpy(points).float()
        
        # 生成BEV表示
        bev_map = self.generate_bev(points_tensor)
        
        # 发布BEV栅格地图
        self.publish_bev(bev_map, msg.header)
        
    def generate_bev(self, points):
        # 过滤点云(只保留感兴趣区域内的点)
        mask_x = (points[:, 0] >= self.x_range[0]) & (points[:, 0] < self.x_range[1])
        mask_y = (points[:, 1] >= self.y_range[0]) & (points[:, 1] < self.y_range[1])
        mask_z = (points[:, 2] >= self.height_threshold[0]) & (points[:, 2] < self.height_threshold[1])
        mask = mask_x & mask_y & mask_z
        points = points[mask]
        
        # 转换点云坐标到BEV网格索引
        x_indices = ((points[:, 0] - self.x_range[0]) / self.resolution).long()
        y_indices = ((points[:, 1] - self.y_range[0]) / self.resolution).long()
        
        # 创建空的BEV图
        bev_map = torch.zeros((self.x_size, self.y_size), dtype=torch.int8)
        
        # 填充BEV图(简单的二值占据表示)
        for i in range(points.shape[0]):
            x_idx, y_idx = x_indices[i], y_indices[i]
            bev_map[x_idx, y_idx] = 100  # 100表示占据(按OccupancyGrid格式)
            
        # 也可以根据高度或密度编码更多信息
        
        return bev_map.numpy()
        
    def publish_bev(self, bev_map, header):
        # 创建并填充OccupancyGrid消息
        grid = OccupancyGrid()
        grid.header = header
        grid.header.frame_id = "map"  # 使用适当的坐标系
        grid.info.resolution = self.resolution
        grid.info.width = self.x_size
        grid.info.height = self.y_size
        grid.info.origin.position.x = self.x_range[0]
        grid.info.origin.position.y = self.y_range[0]
        grid.data = bev_map.flatten().tolist()
        
        # 发布消息
        self.pub_bev.publish(grid)

if __name__ == '__main__':
    rospy.init_node('pointcloud_to_bev')
    converter = PointCloudToBEV()
    rospy.spin()