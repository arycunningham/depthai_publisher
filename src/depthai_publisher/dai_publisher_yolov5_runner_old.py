#!/usr/bin/env python3

'''
Run as:
rosrun depthai_publisher dai_publisher_yolov5_runner
'''
############################### Libraries ###############################
from pathlib import Path
import threading
import csv
import argparse
import time
import sys
import json
import cv2
import numpy as np
import depthai as dai
import rospy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray, String, Bool
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from mavros_msgs.msg import State
import tf2_ros
import tf_conversions
import math
from visualization_msgs.msg import Marker, MarkerArray

############################### Parameters ###############################
pipeline = None
cam_source = 'rgb'
cam = None
syncNN = True
modelsPath = "/home/cdrone/catkin_ws/src/depthai_publisher/src/depthai_publisher/models"
modelName = 'f1'
confJson = 'f1.json'

################################  Yolo Config File
configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = 0.75
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

class DetectedTarget:
    """Class to store detected target information"""
    def __init__(self, target_id, label, confidence, world_x, world_y, world_z, timestamp):
        self.target_id = target_id
        self.label = label
        self.confidence = confidence
        self.world_x = world_x
        self.world_y = world_y
        self.world_z = world_z
        self.timestamp = timestamp
        self.last_seen = timestamp
        
    def distance_to(self, other_target):
        """Calculate 2D distance to another target"""
        dx = self.world_x - other_target.world_x
        dy = self.world_y - other_target.world_y
        return math.sqrt(dx*dx + dy*dy)

class DepthaiCamera():
    fps = 30.0

    pub_topic = '/depthai_node/image/compressed'
    pub_topic_raw = '/depthai_node/image/raw'
    pub_topic_detect = '/depthai_node/detection/compressed'
    pub_topic_cam_inf = '/depthai_node/camera/camera_info'
    
    pub_topic_target_confirmation = '/target_detection/confirmation'
    pub_topic_target_type = '/target_detection/type'
    pub_topic_target_roi = '/target_detection/roi'
    pub_topic_target_list = '/target_detection/target_list'

    def __init__(self):
        self.pipeline = dai.Pipeline()

        if "input_size" in nnConfig:
            self.nn_shape_w, self.nn_shape_h = tuple(map(int, nnConfig.get("input_size").split('x')))

        self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=30)
        self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=30)
        self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=30)
        self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=30)
        
        self.pub_target_confirmation = rospy.Publisher(self.pub_topic_target_confirmation, Bool, queue_size=30)
        self.pub_target_type = rospy.Publisher(self.pub_topic_target_type, String, queue_size=30)
        self.pub_target_roi = rospy.Publisher(self.pub_topic_target_roi, PoseStamped, queue_size=30)
        self.pub_target_list = rospy.Publisher(self.pub_topic_target_list, String, queue_size=30)

        self.sub_marker_id = rospy.Subscriber('/target_detection/marker_id', String, self.aruco_callback)
        self.marker_id = None
        
        self.sub_uav_pose = rospy.Subscriber('/uavasr/pose', PoseStamped, self.callback_uav_pose)
        
        self.current_uav_pose = None
        
        self.detected_targets = []
        self.detected_gating_set = set()  # (label, marker_id) gating
        self.target_id_counter = 0
        self.target_timeout = 10.0
        
        self.camera_offset_x = 0.12
        self.camera_offset_y = 0.0
        self.camera_offset_z = -0.1
        
        self.fx, self.fy = 615.381, 615.381
        self.cx, self.cy = 320.0, 240.0
        
        self.timer = rospy.Timer(rospy.Duration(1.0 / 10), self.publish_camera_info, oneshot=False)

        rospy.loginfo("Publishing images to rostopic: {}".format(self.pub_topic))
        rospy.loginfo("Publishing target detection to: confirmation={}, type={}, roi={}, list={}".format(
            self.pub_topic_target_confirmation, self.pub_topic_target_type, 
            self.pub_topic_target_roi, self.pub_topic_target_list))

        self.br = CvBridge()
        rospy.on_shutdown(lambda: self.shutdown())

    def callback_uav_pose(self, msg):
        """Store current UAV pose for target localization"""
        self.current_uav_pose = msg

    def aruco_callback(self, msg):
        self.marker_id = msg.data

    def publish_camera_info(self, timer=None):
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.height = self.nn_shape_h
        camera_info_msg.width = self.nn_shape_w

        camera_info_msg.K = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        camera_info_msg.D = [-0.10818, 0.12793, 0.00000, 0.00000, -0.04204]
        camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info_msg.P = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.header.stamp = rospy.Time.now()

        self.pub_cam_inf.publish(camera_info_msg)

    def pixel_to_world_coordinates(self, pixel_x, pixel_y, depth_estimate):
        """Convert pixel coordinates to world coordinates using UAV pose"""
        if self.current_uav_pose is None:
            rospy.logwarn("UAV pose not available for target localization")
            return None
            
        x_norm = (pixel_x - self.cx) / self.fx
        y_norm = (pixel_y - self.cy) / self.fy
        
        x_cam = x_norm * depth_estimate
        y_cam = y_norm * depth_estimate
        z_cam = -depth_estimate
        
        uav_pos = self.current_uav_pose.pose.position
        uav_orient = self.current_uav_pose.pose.orientation
        
        yaw = math.atan2(2.0 * (uav_orient.w * uav_orient.z + uav_orient.x * uav_orient.y),
                        1.0 - 2.0 * (uav_orient.y**2 + uav_orient.z**2))
        
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        camera_world_x = uav_pos.x + cos_yaw * self.camera_offset_x - sin_yaw * self.camera_offset_y
        camera_world_y = uav_pos.y + sin_yaw * self.camera_offset_x + cos_yaw * self.camera_offset_y
        camera_world_z = uav_pos.z + self.camera_offset_z
        
        target_x_local = cos_yaw * x_cam - sin_yaw * y_cam
        target_y_local = sin_yaw * x_cam + cos_yaw * y_cam
        
        world_x = camera_world_x + target_x_local
        world_y = camera_world_y + target_y_local
        world_z = camera_world_z + z_cam
        
        if world_z < 0:
            rospy.logwarn("Target below ground level (Z={:.2f}), adjusting to Z=0".format(world_z))
            world_z = 0.0
        
        return world_x, world_y, world_z

    def update_target_list(self, new_detections, timestamp):
        """Update the list of detected targets"""
        current_time_sec = timestamp.to_sec()
        
        self.detected_targets = [
            target for target in self.detected_targets
            if (current_time_sec - target.last_seen.to_sec()) < self.target_timeout
        ]
        
        for detection in new_detections:
            # Gating principle: avoid duplicate detections
            detection_key = (detection.label, self.marker_id)
            if detection_key in self.detected_gating_set:
                rospy.loginfo(f"[GATE] Duplicate detection ignored: {detection_key}")
                continue
            else:
                self.detected_gating_set.add(detection_key)
                rospy.loginfo(f"[GATE] New detection accepted: {detection_key}")

            center_x = (detection.xmin + detection.xmax) / 2.0 * self.nn_shape_w
            center_y = (detection.ymin + detection.ymax) / 2.0 * self.nn_shape_h
            
            bbox_width = (detection.xmax - detection.xmin) * self.nn_shape_w
            bbox_height = (detection.ymax - detection.ymin) * self.nn_shape_h
            bbox_area = bbox_width * bbox_height
            max_area = self.nn_shape_w * self.nn_shape_h
            
            depth_estimate = max(0.5, 5.0 * (1.0 - bbox_area / max_area))
            
            world_coords = self.pixel_to_world_coordinates(center_x, center_y, depth_estimate)
            
            if world_coords is None:
                continue
                
            world_x, world_y, world_z = world_coords
            
            if world_z < 0:
                rospy.loginfo("Detection skipped: Below ground")
                continue
            
            new_target = DetectedTarget(
                self.target_id_counter, labels[detection.label], detection.confidence,
                world_x, world_y, world_z, timestamp
            )
            
            self.detected_targets.append(new_target)
            self.target_id_counter += 1
            rospy.loginfo("New target added: ID={} {} at [{:.2f}, {:.2f}, {:.2f}] confidence: {:.2f}".format(
                new_target.target_id, new_target.label, world_x, world_y, world_z, detection.confidence))
            
            roi_msg = PoseStamped()
            roi_msg.header.stamp = timestamp
            roi_msg.header.frame_id = "map"
        
            roi_msg.pose.position.x = world_x
            roi_msg.pose.position.y = world_y
            roi_msg.pose.position.z = world_z
        
            roi_msg.pose.orientation.w = 1.0
            roi_msg.pose.orientation.x = 0.0
            roi_msg.pose.orientation.y = 0.0
            roi_msg.pose.orientation.z = 0.0

            self.pub_target_roi.publish(roi_msg)

    def publish_target_list(self, timestamp):
        """Publish the current list of detected targets"""
        if len(self.detected_targets) == 0:
            target_list_msg = String()
            target_list_msg.data = "No targets detected"
            self.pub_target_list.publish(target_list_msg)
            return
        
        target_info = []
        for target in self.detected_targets:
            target_info.append("ID:{} Type:{} Pos:[{:.2f},{:.2f},{:.2f}] Conf:{:.2f}".format(
                target.target_id, target.label, 
                target.world_x, target.world_y, target.world_z,
                target.confidence
            ))
        
        target_list_msg = String()
        target_list_msg.data = "Targets: " + " | ".join(target_info)
        self.pub_target_list.publish(target_list_msg)

    def publish_target_detection(self, detections, timestamp):
        """Publish target detection data compatible with spar/breadcrumb system"""
        if len(detections) == 0:
            confirmation_msg = Bool()
            confirmation_msg.data = False
            self.pub_target_confirmation.publish(confirmation_msg)
            self.publish_target_list(timestamp)
            return
        
        self.update_target_list(detections, timestamp)
        self.publish_target_list(timestamp)
        
        best_detection = max(detections, key=lambda d: d.confidence)
        
        center_x = (best_detection.xmin + best_detection.xmax) / 2.0 * self.nn_shape_w
        center_y = (best_detection.ymin + best_detection.ymax) / 2.0 * self.nn_shape_h
        
        bbox_width = (best_detection.xmax - best_detection.xmin) * self.nn_shape_w
        bbox_height = (best_detection.ymax - best_detection.ymin) * self.nn_shape_h
        bbox_area = bbox_width * bbox_height
        max_area = self.nn_shape_w * self.nn_shape_h
        
        depth_estimate = max(0.5, 5.0 * (1.0 - bbox_area / max_area))
        
        world_coords = self.pixel_to_world_coordinates(center_x, center_y, depth_estimate)
        
        if world_coords is None:
            rospy.logwarn("Cannot localize target - UAV pose unavailable")
            return
            
        world_x, world_y, world_z = world_coords
        
        confirmation_msg = Bool()
        confirmation_msg.data = True
        self.pub_target_confirmation.publish(confirmation_msg)
        
        type_msg = String()
        type_msg.data = labels[best_detection.label]
        if type_msg.data == "marker" and (self.marker_id is not None):
            type_msg.data = "marker_{}".format(str(self.marker_id))
        self.pub_target_type.publish(type_msg)

        roi_msg = PoseStamped()
        roi_msg.header.stamp = timestamp
        roi_msg.header.frame_id = "map"
        roi_msg.pose.position.x = world_x
        roi_msg.pose.position.y = world_y
        roi_msg.pose.position.z = world_z
        roi_msg.pose.orientation.w = 1.0
        roi_msg.pose.orientation.x = 0.0
        roi_msg.pose.orientation.y = 0.0
        roi_msg.pose.orientation.z = 0.0
        
        self.pub_target_roi.publish(roi_msg)
        
        rospy.loginfo("Best target: {} at world coords [{:.2f}, {:.2f}, {:.2f}] confidence: {:.2f}".format(
            labels[best_detection.label], world_x, world_y, world_z, best_detection.confidence))
        rospy.loginfo("Total unique targets tracked: {}".format(len(self.detected_targets)))

    def run(self):
        modelPathName = f'{modelsPath}/{modelName}/{modelName}.blob'
        print(metadata)
        nnPath = str((Path(__file__).parent / Path(modelPathName)).resolve().absolute())
        print(nnPath)

        pipeline = self.createPipeline(nnPath)

        with dai.Device() as device:
            cams = device.getConnectedCameras()
            if cam_source != "rgb":
                raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
            device.startPipeline(pipeline)

            q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            detections = []
            start_time = time.time()
            counter = 0
            fps = 0

            while True:
                found_classes = []
                inRgb = q_nn_input.get()
                inDet = q_nn.get()

                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                else:
                    print("Cam Image empty, trying again...")
                    continue
                
                current_time = rospy.Time.now()
                
                if inDet is not None:
                    detections = inDet.detections
                    for detection in detections:
                        rospy.loginfo("{},{},{},{},{},{}".format(labels[detection.label],detection.confidence,detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        found_classes.append(detection.label)
                    found_classes = np.unique(found_classes)
                    overlay = self.show_yolo(frame, detections)
                    
                    self.publish_target_detection(detections, current_time)
                        
                else:
                    print("Detection empty, trying again...")
                    confirmation_msg = Bool()
                    confirmation_msg.data = False
                    self.pub_target_confirmation.publish(confirmation_msg)
                    self.publish_target_list(current_time)
                    continue

                if frame is not None:
                    cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    cv2.putText(overlay, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    cv2.putText(overlay, "Tracked targets: {}".format(len(self.detected_targets)), (2, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    self.publish_to_ros(frame)
                    self.publish_detect_to_ros(overlay)
                    self.publish_camera_info()

                counter += 1
                if (time.time() - start_time) > 1:
                    fps = counter / (time.time() - start_time)
                    counter = 0
                    start_time = time.time()

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "camera_frame"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image.publish(msg_out)
        
        msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
        msg_img_raw.header.frame_id = "camera_frame"
        self.pub_image_raw.publish(msg_img_raw)

    def publish_detect_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "camera_frame"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image_detect.publish(msg_out)

    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def show_yolo(self, frame, detections):
        color = (255, 0, 0)
        overlay = frame.copy()
        for detection in detections:
            bbox = self.frameNorm(overlay, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(overlay, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(overlay, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return overlay

    def createPipeline(self, nnPath):
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2022_1)

        detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        detection_nn.setConfidenceThreshold(confidenceThreshold)
        detection_nn.setNumClasses(classes)
        detection_nn.setCoordinateSize(coordinates)
        detection_nn.setAnchors(anchors)
        detection_nn.setAnchorMasks(anchorMasks)
        detection_nn.setIouThreshold(iouThreshold)
        detection_nn.setBlobPath(nnPath)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)

        if cam_source == 'rgb':
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(self.nn_shape_w, self.nn_shape_h)
            cam.setInterleaved(False)
            cam.preview.link(detection_nn.input)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setFps(40)
            print("Using RGB camera...")
        elif cam_source == 'left':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
            print("Using BW Left cam")
        elif cam_source == 'right':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            print("Using BW Right cam")

        if cam_source != 'rgb':
            manip = pipeline.create(dai.node.ImageManip)
            manip.setResize(self.nn_shape_w, self.nn_shape_h)
            manip.setKeepAspectRatio(True)
            manip.setFrameType(dai.RawImgFrame.Type.RGB888p)
            cam.out.link(manip.inputImage)
            manip.out.link(detection_nn.input)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("nn_input")
        xout_rgb.input.setBlocking(False)
        detection_nn.passthrough.link(xout_rgb.input)

        xinDet = pipeline.create(dai.node.XLinkOut)
        xinDet.setStreamName("nn")
        xinDet.input.setBlocking(False)
        detection_nn.out.link(xinDet.input)

        return pipeline

    def shutdown(self):
        cv2.destroyAllWindows()

def main():
    rospy.init_node('depthai_node')
    dai_cam = DepthaiCamera()

    while not rospy.is_shutdown():
        dai_cam.run()

    dai_cam.shutdown()

if __name__ == '__main__':
    main()