#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import time

def clear_tf():
    rospy.init_node('tf_clearer', anonymous=True)
    
    # 创建一个TF广播器
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    
    # 创建一个空变换
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = "world"
    transform.child_frame_id = "base_link_clear"
    transform.transform.translation.x = 0.0
    transform.transform.translation.y = 0.0
    transform.transform.translation.z = 0.0
    transform.transform.rotation.w = 1.0
    
    # 广播这个变换
    broadcaster.sendTransform(transform)
    
    # 等待一会儿确保变换被处理
    time.sleep(1.0)
    
    rospy.loginfo("TF cleared!")

if __name__ == '__main__':
    try:
        clear_tf()
    except rospy.ROSInterruptException:
        pass