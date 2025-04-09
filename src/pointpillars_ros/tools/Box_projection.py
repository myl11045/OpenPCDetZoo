import torch
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

class BoxProjection:
    def __init__(self):
        # 创建检测框发布者
        self.pub_boxes = rospy.Publisher('/bev_boxes', MarkerArray, queue_size=1)
        
    def project_boxes_to_bev(self, boxes_3d):
        """
        将3D检测框投影到BEV平面
        boxes_3d: torch.Tensor, 形状 [N, 7], 每行为 [x, y, z, l, w, h, heading]
        返回: 2D框的顶点坐标 [N, 4, 2]
        """
        # 每个3D框的中心点
        centers = boxes_3d[:, :2]
        
        # 框的尺寸
        dims = boxes_3d[:, 3:5]  # 只取长和宽
        
        # 朝向角
        headings = boxes_3d[:, 6]
        
        # 计算未旋转的框的四个角点(相对于中心)
        l, w = dims[:, 0:1], dims[:, 1:2]
        corners_norm = torch.tensor([
            [1, 1], [1, -1], [-1, -1], [-1, 1]
        ], dtype=torch.float32) * 0.5  # 单位化的角点
        
        # 根据框的尺寸缩放角点
        corners = torch.cat([
            corners_norm[:, 0:1] * l.unsqueeze(1),
            corners_norm[:, 1:2] * w.unsqueeze(1)
        ], dim=-1)  # [N, 4, 2]
        
        # 应用旋转
        cos_rots = torch.cos(headings).view(-1, 1, 1)
        sin_rots = torch.sin(headings).view(-1, 1, 1)
        
        # 旋转矩阵
        ones = torch.ones_like(cos_rots)
        zeros = torch.zeros_like(cos_rots)
        
        # [cos, -sin; sin, cos]
        rot_matrix = torch.cat([
            torch.cat([cos_rots, -sin_rots], dim=-1),
            torch.cat([sin_rots, cos_rots], dim=-1)
        ], dim=1)  # [N, 2, 2]
        
        # 应用旋转
        corners_rotated = torch.matmul(corners, rot_matrix.transpose(1, 2))
        
        # 平移到世界坐标
        corners_global = corners_rotated + centers.unsqueeze(1)
        
        return corners_global
        
    def publish_box_markers(self, corners, header, ns="bev_boxes"):
        """
        将BEV投影的检测框发布为MarkerArray用于RViz可视化
        corners: [N, 4, 2] 每个框的四个角点
        """
        marker_array = MarkerArray()
        
        for i, box_corners in enumerate(corners):
            # 创建线条Marker
            marker = Marker()
            marker.header = header
            marker.ns = ns
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.1  # 线宽
            
            # 设置颜色(这里使用蓝色)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            
            # 添加角点(回到第一个点形成闭环)
            for corner in box_corners:
                p = Point()
                p.x = float(corner[0])
                p.y = float(corner[1])
                p.z = 0.0  # BEV平面的z值设为0
                marker.points.append(p)
            
            # 添加第一个点以闭合线条
            p = Point()
            p.x = float(box_corners[0][0])
            p.y = float(box_corners[0][1])
            p.z = 0.0
            marker.points.append(p)
            
            marker_array.markers.append(marker)
        
        self.pub_boxes.publish(marker_array)