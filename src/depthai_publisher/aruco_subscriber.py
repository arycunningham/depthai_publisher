#!/usr/bin/env python3

import cv2
import math
import numpy as np

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Bool, String
from mavros_msgs.msg import State

class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
    aruco_params = cv2.aruco.DetectorParameters_create()

    frame_sub_topic = '/depthai_node/image/compressed'
    
    # Target detection topics for spar/breadcrumb integration (same as YOLOv5)
    pub_topic_target_confirmation = '/target_detection/confirmation'
    pub_topic_target_type = '/target_detection/type'
    pub_topic_target_roi = '/target_detection/roi'

    def __init__(self):
        # Publishers for processed ArUco image
        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=30)

        # Publishers for target detection (compatible with spar/breadcrumb system)
        self.pub_target_confirmation = rospy.Publisher(self.pub_topic_target_confirmation, Bool, queue_size=10)
        self.pub_target_type = rospy.Publisher(self.pub_topic_target_type, String, queue_size=10)
        self.pub_target_roi = rospy.Publisher(self.pub_topic_target_roi, PoseStamped, queue_size=10)
        
        # Subscribe to UAV pose from MAVROS
        # self.sub_uav_pose = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.callback_uav_pose)
        # Subscribe to UAV Emulated pose
        self.sub_uav_pose = rospy.Subscriber('/uavasr/pose', PoseStamped, self.callback_uav_pose)
        
        # Subscribe to camera info for intrinsics
        self.sub_camera_info = rospy.Subscriber('/depthai_node/camera/camera_info', CameraInfo, self.callback_camera_info)
        
        # UAV pose and camera parameters storage
        self.current_uav_pose = None
        self.camera_info = None
        
        # Camera parameters (will be updated from camera_info)
        self.fx, self.fy = 615.381, 615.381
        self.cx, self.cy = 320.0, 240.0
        
        # Camera offset from UAV center (matching tf2_broadcaster_frames and YOLOv5)
        self.camera_offset_x = 0.12   # Forward
        self.camera_offset_y = 0.0   # Right  
        self.camera_offset_z = -0.1 # Down
        
        # ArUco marker size in meters (adjust based on your actual markers)
        self.marker_size = 0.1  # 10cm markers
        
        rospy.loginfo("ArUco detector with localization initialized")
        rospy.loginfo("Publishing target detection to: confirmation={}, type={}, roi={}".format(
            self.pub_topic_target_confirmation, self.pub_topic_target_type, self.pub_topic_target_roi))

        self.br = CvBridge()

        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback)

    def callback_uav_pose(self, msg):
        """Store current UAV pose for target localization"""
        self.current_uav_pose = msg

    def callback_camera_info(self, msg):
        """Update camera intrinsics from camera_info"""
        self.camera_info = msg
        if len(msg.K) >= 9:
            self.fx = msg.K[0]
            self.fy = msg.K[4] 
            self.cx = msg.K[2]
            self.cy = msg.K[5]

    def estimate_marker_distance(self, corners):
        """
        Estimate distance to ArUco marker based on its apparent size
        """
        if len(corners) == 0:
            return 2.0  # Default distance
            
        # Calculate marker area in pixels
        corner_points = corners[0].reshape((4, 2))
        
        # Calculate bounding box
        x_coords = corner_points[:, 0]
        y_coords = corner_points[:, 1]
        
        width_pixels = np.max(x_coords) - np.min(x_coords)
        height_pixels = np.max(y_coords) - np.min(y_coords)
        
        # Use average of width and height for size estimation
        avg_size_pixels = (width_pixels + height_pixels) / 2.0
        
        # Estimate distance based on known marker size and focal length
        if avg_size_pixels > 0:
            # Using similar triangles: distance = (real_size * focal_length) / pixel_size
            distance = (self.marker_size * self.fx) / avg_size_pixels
            return max(0.5, min(distance, 10.0))  # Clamp between 0.5m and 10m
        
        return 2.0  # Default fallback

    def pixel_to_world_coordinates(self, pixel_x, pixel_y, depth_estimate):
        """
        Convert pixel coordinates to world coordinates using UAV pose
        Returns world coordinates (x, y, z) or None if UAV pose unavailable
        """
        if self.current_uav_pose is None:
            rospy.logwarn("UAV pose not available for ArUco target localization")
            return None
            
        # Convert pixel to normalized camera coordinates
        x_norm = (pixel_x - self.cx) / self.fx
        y_norm = (pixel_y - self.cy) / self.fy
        
        # Camera frame coordinates (camera pointing down)
        # Assuming camera is pointing downward (-Z axis)
        x_cam = x_norm * depth_estimate
        y_cam = y_norm * depth_estimate
        z_cam = -depth_estimate  # Negative because camera points down
        
        # UAV position and orientation
        uav_pos = self.current_uav_pose.pose.position
        uav_orient = self.current_uav_pose.pose.orientation
        
        # Extract yaw from quaternion (simplified for level flight)
        yaw = math.atan2(2.0 * (uav_orient.w * uav_orient.z + uav_orient.x * uav_orient.y),
                        1.0 - 2.0 * (uav_orient.y**2 + uav_orient.z**2))
        
        # Apply UAV rotation and camera offset to get world coordinates
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        # Camera position in world frame
        camera_world_x = uav_pos.x + cos_yaw * self.camera_offset_x - sin_yaw * self.camera_offset_y
        camera_world_y = uav_pos.y + sin_yaw * self.camera_offset_x + cos_yaw * self.camera_offset_y
        camera_world_z = uav_pos.z + self.camera_offset_z
        
        # Target position in world frame
        # Rotate camera coordinates by UAV yaw
        target_x_local = cos_yaw * x_cam - sin_yaw * y_cam
        target_y_local = sin_yaw * x_cam + cos_yaw * y_cam
        
        world_x = camera_world_x + target_x_local
        world_y = camera_world_y + target_y_local
        world_z = camera_world_z + z_cam
        
        return world_x, world_y, world_z

    def publish_aruco_detection(self, marker_ids, marker_corners, timestamp):
        """
        Publish ArUco detection data compatible with spar/breadcrumb system
        """
        if len(marker_ids) == 0:
            # Publish no target found
            confirmation_msg = Bool()
            confirmation_msg.data = False
            self.pub_target_confirmation.publish(confirmation_msg)
            return
            
        # Take the first detected marker (or you could prioritize by ID)
        marker_id = marker_ids[0]
        corners = marker_corners[0]
        
        # Calculate marker center in pixels
        corner_points = corners.reshape((4, 2))
        center_x = np.mean(corner_points[:, 0])
        center_y = np.mean(corner_points[:, 1])
        
        # Estimate distance to marker
        depth_estimate = self.estimate_marker_distance([corners])
        
        # Convert to world coordinates
        world_coords = self.pixel_to_world_coordinates(center_x, center_y, depth_estimate)
        
        if world_coords is None:
            rospy.logwarn("Cannot localize ArUco marker - UAV pose unavailable")
            return
            
        world_x, world_y, world_z = world_coords
        
        # Publish target confirmation
        confirmation_msg = Bool()
        confirmation_msg.data = True
        self.pub_target_confirmation.publish(confirmation_msg)
        
        # Publish target type (ArUco marker ID)
        type_msg = String()
        type_msg.data = f"ArUco_{marker_id}"
        self.pub_target_type.publish(type_msg)
        
        # Publish target ROI (compatible with spar ROI subscriber)
        roi_msg = PoseStamped()
        roi_msg.header.stamp = timestamp
        roi_msg.header.frame_id = "map"  # World coordinate frame
        
        roi_msg.pose.position.x = world_x
        roi_msg.pose.position.y = world_y
        roi_msg.pose.position.z = world_z
        
        # Identity orientation
        roi_msg.pose.orientation.w = 1.0
        roi_msg.pose.orientation.x = 0.0
        roi_msg.pose.orientation.y = 0.0
        roi_msg.pose.orientation.z = 0.0
        
        self.pub_target_roi.publish(roi_msg)
        
        rospy.loginfo("ArUco marker {} detected at world coords [{:.2f}, {:.2f}, {:.2f}] distance: {:.2f}m".format(
            marker_id, world_x, world_y, world_z, depth_estimate))

    def img_callback(self, msg_in):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg_in)
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        timestamp = msg_in.header.stamp

        # Find ArUco markers
        aruco_annotated = self.find_aruco(frame)
        
        # Detect markers for localization
        (corners, ids, _) = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)
        
        # Publish detection data for spar/breadcrumb integration
        if ids is not None:
            self.publish_aruco_detection(ids.flatten(), corners, timestamp)
        else:
            # Publish no target found
            confirmation_msg = Bool()
            confirmation_msg.data = False
            self.pub_target_confirmation.publish(confirmation_msg)
        
        # Publish annotated image
        self.publish_to_ros(aruco_annotated)

    def find_aruco(self, frame):
        """
        Detect ArUco markers and annotate the frame
        """
        (corners, ids, _) = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)

        if len(corners) > 0:
            ids = ids.flatten()

            for (marker_corner, marker_ID) in zip(corners, ids):
                corners_reshaped = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners_reshaped

                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                # Draw marker outline
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                # Calculate center for distance estimation
                center_x = int(np.mean([top_left[0], top_right[0], bottom_right[0], bottom_left[0]]))
                center_y = int(np.mean([top_left[1], top_right[1], bottom_right[1], bottom_left[1]]))
                
                # Estimate distance
                distance = self.estimate_marker_distance([marker_corner])

                rospy.loginfo("ArUco detected, ID: {}, Distance: {:.2f}m".format(marker_ID, distance))

                # Draw marker ID and distance
                cv2.putText(frame, f"ID:{marker_ID}", 
                           (top_left[0], top_left[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"{distance:.1f}m", 
                           (top_left[0], top_left[1] - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        return frame

    def publish_to_ros(self, frame):
        """Publish the annotated ArUco frame"""
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.header.frame_id = "camera_frame"
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

        self.aruco_pub.publish(msg_out)


def main():
    rospy.init_node('aruco_detector_with_localization', anonymous=True)
    rospy.loginfo("ArUco detector with localization processing images...")

    aruco_detect = ArucoDetector()

    rospy.spin()

if __name__ == '__main__':
    main()