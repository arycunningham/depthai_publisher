#!/usr/bin/env python3

'''
Run as:
rosrun depthai_publisher dai_publisher_yolov5_runner
'''

############################### Libraries ###############################
from pathlib import Path
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
from std_msgs.msg import Float32MultiArray, String, Bool, Header
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from mavros_msgs.msg import State
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
# confidenceThreshold = metadata.get("confidence_threshold", {})
confidenceThreshold = 0.75
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", [])

class DetectedTarget:
    """Store detected target information"""
    def __init__(self, target_id, label, confidence, world_x, world_y, world_z, timestamp, marker_id_str=None):
        self.target_id = target_id
        self.label = label
        self.confidence = confidence
        self.world_x = world_x
        self.world_y = world_y
        self.world_z = world_z
        self.timestamp = timestamp
        self.last_seen = timestamp
        self.marker_id_str = marker_id_str  # Store the full marker ID string for RViz

    def distance_to(self, other_target):
        dx = self.world_x - other_target.world_x
        dy = self.world_y - other_target.world_y
        return math.sqrt(dx*dx + dy*dy)

class DepthaiCamera():
    fps = 30.0

    pub_topic = '/depthai_node/image/compressed'
    pub_topic_raw = '/depthai_node/image/raw'
    pub_topic_detect = '/depthai_node/detection/compressed'
    pub_topic_cam_inf = '/depthai_node/camera/camera_info'

    # Target detection topics (SPaR/Breadcrumb compatibility)
    pub_topic_target_confirmation = '/target_detection/confirmation'
    pub_topic_target_type = '/target_detection/type'
    pub_topic_target_roi = '/target_detection/roi'
    pub_topic_target_list = '/target_detection/target_list'

    # Marker topics
    pub_topic_marker = '/target_detection/visual_marker'
    pub_topic_marker_array = '/target_detection/visual_marker_array'

    def __init__(self):
        self.pipeline = dai.Pipeline()

        if "input_size" in nnConfig:
            self.nn_shape_w, self.nn_shape_h = tuple(map(int, nnConfig.get("input_size").split('x')))
        else:
            # Default to a common preview size if missing
            self.nn_shape_w, self.nn_shape_h = 640, 360

        # Image publishers
        self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=30)
        self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=30)
        self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=30)
        self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=30, latch=True)

        # Target pubs
        self.pub_target_confirmation = rospy.Publisher(self.pub_topic_target_confirmation, Bool, queue_size=30, latch=True)
        self.pub_target_type = rospy.Publisher(self.pub_topic_target_type, String, queue_size=30, latch=True)
        self.pub_target_roi = rospy.Publisher(self.pub_topic_target_roi, PoseStamped, queue_size=30)
        self.pub_target_list = rospy.Publisher(self.pub_topic_target_list, String, queue_size=30)

        # Marker I/O
        self.sub_marker_id = rospy.Subscriber('/target_detection/marker_id', String, self.aruco_callback)
        self.marker_id = None

        # Latching for RViz so last state persists
        self.pub_marker = rospy.Publisher(self.pub_topic_marker, Marker, queue_size=10, latch=True)
        self.pub_marker_array = rospy.Publisher(self.pub_topic_marker_array, MarkerArray, queue_size=10, latch=True)

        # UAV pose in map frame
        self.sub_uav_pose = rospy.Subscriber('/uavasr/pose', PoseStamped, self.callback_uav_pose)
        self.current_uav_pose = None

        # ROI diversion status from guidance
        self.sub_performing_roi = rospy.Subscriber('/guidance/performing_roi', Bool, self.callback_performing_roi)
        self.guidance_performing_roi = False

        # Tracking
        self.detected_targets = []
        self.detected_gating_set = set()  # (label, marker_id) gating for markers only
        self.target_id_counter = 0
        self.target_timeout = 10.0  # seconds
        self.type_instance_counters = {}  # Track instance numbers per type (e.g., {'fire': 1, 'smoke': 2})
        self.min_detection_interval = 2.5  # Minimum seconds between detections
        self.min_spatial_distance = 0.15  # Minimum meters between targets to avoid duplicates

        # Camera extrinsics (relative to UAV base)
        self.camera_offset_x = 0.12
        self.camera_offset_y = 0.0
        self.camera_offset_z = -0.1

        # Intrinsics (approx)
        self.fx, self.fy = 615.381, 615.381
        self.cx, self.cy = 320.0, 240.0

        # Publish CameraInfo periodically
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_camera_info, oneshot=False)

        rospy.loginfo("Publishing images to: {}".format(self.pub_topic))
        rospy.loginfo("Target detection topics: confirmation={}, type={}, roi={}, list={}".format(
            self.pub_topic_target_confirmation, self.pub_topic_target_type,
            self.pub_topic_target_roi, self.pub_topic_target_list))
        rospy.loginfo("Marker topics: marker={}, array={}".format(self.pub_topic_marker, self.pub_topic_marker_array))

        self.br = CvBridge()
        rospy.on_shutdown(self.shutdown)

    def callback_uav_pose(self, msg):
        """Store current UAV pose for world coordinate transforms."""
        self.current_uav_pose = msg

    def callback_performing_roi(self, msg):
        """Store whether guidance is currently performing an ROI diversion."""
        self.guidance_performing_roi = msg.data
        if self.guidance_performing_roi:
            rospy.loginfo("Guidance is performing ROI diversion - detection gating active")
        else:
            rospy.loginfo("Guidance resumed normal flight - detection gating released")

    def aruco_callback(self, msg):
        """Store detected ArUco marker ID for labeling."""
        self.marker_id = msg.data

    def publish_camera_info(self, _timer_event=None):
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.height = self.nn_shape_h
        camera_info_msg.width = self.nn_shape_w

        camera_info_msg.K = [self.fx, 0.0, self.cx,
                             0.0, self.fy, self.cy,
                             0.0, 0.0, 1.0]
        camera_info_msg.D = [-0.10818, 0.12793, 0.00000, 0.00000, -0.04204]
        camera_info_msg.R = [1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0]
        camera_info_msg.P = [self.fx, 0.0, self.cx, 0.0,
                             0.0, self.fy, self.cy, 0.0,
                             0.0, 0.0, 1.0, 0.0]
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.header.stamp = rospy.Time.now()
        self.pub_cam_inf.publish(camera_info_msg)

    def pixel_to_world_coordinates(self, pixel_x, pixel_y, depth_estimate):
        """Project pixel to a rough world position using UAV pose + simple pinhole & yaw-only rotation."""
        if self.current_uav_pose is None:
            rospy.logwarn("UAV pose not available for target localization")
            return None

        # Normalize pixel
        x_norm = (pixel_x - self.cx) / self.fx
        y_norm = (pixel_y - self.cy) / self.fy

        # Camera forward points -z in this convention
        x_cam = x_norm * depth_estimate
        y_cam = y_norm * depth_estimate
        z_cam = -depth_estimate

        uav_pos = self.current_uav_pose.pose.position
        uav_orient = self.current_uav_pose.pose.orientation

        # Approx yaw from quaternion
        yaw = math.atan2(2.0 * (uav_orient.w * uav_orient.z + uav_orient.x * uav_orient.y),
                         1.0 - 2.0 * (uav_orient.y**2 + uav_orient.z**2))

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # Camera origin in world
        camera_world_x = uav_pos.x + cos_yaw * self.camera_offset_x - sin_yaw * self.camera_offset_y
        camera_world_y = uav_pos.y + sin_yaw * self.camera_offset_x + cos_yaw * self.camera_offset_y
        camera_world_z = uav_pos.z + self.camera_offset_z

        # Rotate camera-frame XY into world-frame XY using yaw
        target_x_local = cos_yaw * x_cam - sin_yaw * y_cam
        target_y_local = sin_yaw * x_cam + cos_yaw * y_cam

        world_x = camera_world_x + target_x_local
        world_y = camera_world_y + target_y_local
        world_z = camera_world_z + z_cam

        if world_z < 0:
            rospy.logwarn("Target below ground level (Z={:.2f}), clamping to 0".format(world_z))
            world_z = 0.0

        return world_x, world_y, world_z

    def update_target_list(self, new_detections, timestamp):
        """Refresh tracked targets with gating to avoid duplicate spam."""
        current_time_sec = timestamp.to_sec()

        # Remove stale targets
        self.detected_targets = [
            t for t in self.detected_targets
            if (current_time_sec - t.last_seen.to_sec()) < self.target_timeout
        ]

        for detection in new_detections:
            # Gate by (label, marker_id) to reduce duplicates
            label_str = labels[detection.label] if detection.label < len(labels) else str(detection.label)
            
            # For marker detections, wait for valid marker ID and use strict gating
            if label_str == "marker":
                if self.marker_id is None or self.marker_id == "":
                    rospy.logwarn("[GATE] Marker detected but no ArUco ID available yet, skipping...")
                    continue
                gate_key = (label_str, self.marker_id)
                marker_id_str = "marker_{}".format(self.marker_id)
                
                # Strict gating for markers - only allow one per ArUco ID
                if gate_key in self.detected_gating_set:
                    rospy.loginfo(f"[GATE] Duplicate marker ignored: {gate_key}")
                    continue
                
                self.detected_gating_set.add(gate_key)
            else:
                # For smoke/fire, allow multiple instances but use spatial + temporal gating
                marker_id_str = None
            
            rospy.loginfo(f"[GATE] Processing: {label_str}")

            # Calculate bounding box center and area for depth estimation
            center_x = (detection.xmin + detection.xmax) / 2.0 * self.nn_shape_w
            center_y = (detection.ymin + detection.ymax) / 2.0 * self.nn_shape_h
            bbox_w = (detection.xmax - detection.xmin) * self.nn_shape_w
            bbox_h = (detection.ymax - detection.ymin) * self.nn_shape_h
            bbox_area = bbox_w * bbox_h
            max_area = self.nn_shape_w * self.nn_shape_h

            # Crude depth heuristic (bigger box → closer)
            depth_estimate = max(0.5, 5.0 * (1.0 - bbox_area / max_area))

            world_coords = self.pixel_to_world_coordinates(center_x, center_y, depth_estimate)
            if world_coords is None:
                continue
            
            world_x, world_y, world_z = world_coords

            # Spatial and temporal gating for non-marker targets (smoke, fire, etc.)
            if label_str != "marker":
                should_skip = False
                
                # Block all new detections if guidance is performing an ROI diversion
                if self.guidance_performing_roi:
                    rospy.loginfo(f"[GATE] {label_str} blocked - guidance is performing ROI diversion")
                    continue
                
                for existing_target in self.detected_targets:
                    # Only check against same type
                    if existing_target.label != label_str:
                        continue
                    
                    # Calculate spatial distance (2D for ground-level targets)
                    dx = world_x - existing_target.world_x
                    dy = world_y - existing_target.world_y
                    spatial_distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Calculate time since this target was first detected
                    time_since_detection = current_time_sec - existing_target.timestamp.to_sec()
                    
                    # Gate if too close spatially OR too recent temporally
                    if spatial_distance < self.min_spatial_distance:
                        rospy.loginfo(f"[GATE] {label_str} too close to existing target ({spatial_distance:.2f}m < {self.min_spatial_distance}m), skipping")
                        should_skip = True
                        break
                    
                    if time_since_detection < self.min_detection_interval:
                        rospy.loginfo(f"[GATE] {label_str} detected too soon after previous ({time_since_detection:.2f}s < {self.min_detection_interval}s), skipping")
                        should_skip = True
                        break
                
                if should_skip:
                    continue

            # Generate unique target ID string based on type
            if marker_id_str is not None:
                # For markers, use "marker_#"
                target_id_str = marker_id_str
            else:
                # For other types, use type with instance number
                if label_str not in self.type_instance_counters:
                    self.type_instance_counters[label_str] = 0
                self.type_instance_counters[label_str] += 1
                instance_num = self.type_instance_counters[label_str]
                target_id_str = "{}_{}".format(label_str, instance_num)

            # Create and store new target
            new_target = DetectedTarget(
                self.target_id_counter, label_str, detection.confidence,
                world_x, world_y, world_z, timestamp, marker_id_str=target_id_str
            )
            self.detected_targets.append(new_target)
            self.target_id_counter += 1

            rospy.loginfo("New target ID={} ({}) {} @ [{:.2f}, {:.2f}, {:.2f}] conf:{:.2f}".format(
                new_target.target_id, target_id_str, new_target.label, world_x, world_y, world_z, detection.confidence))

            # Publish individual ROI for this target
            self._publish_single_roi(world_x, world_y, world_z, timestamp)

        # After processing all detections, publish consolidated marker array
        self.publish_marker_array_snapshot(self.detected_targets)

    def publish_target_list(self, timestamp):
        """Publish human-readable string of all tracked targets."""
        if len(self.detected_targets) == 0:
            msg = String(data="No targets detected")
            self.pub_target_list.publish(msg)
            return

        items = []
        for t in self.detected_targets:
            items.append("ID:{} ({}) Type:{} Pos:[{:.2f},{:.2f},{:.2f}] Conf:{:.2f}".format(
                t.target_id, t.marker_id_str, t.label, t.world_x, t.world_y, t.world_z, t.confidence))
        self.pub_target_list.publish(String(data="Targets: " + " | ".join(items)))

    def _publish_single_roi(self, world_x, world_y, world_z, timestamp):
        """Helper to publish a single ROI pose."""
        roi_msg = PoseStamped()
        roi_msg.header.stamp = timestamp
        roi_msg.header.frame_id = "map"
        roi_msg.pose.position.x = world_x
        roi_msg.pose.position.y = world_y
        roi_msg.pose.position.z = world_z
        roi_msg.pose.orientation.w = 1.0
        self.pub_target_roi.publish(roi_msg)

    def publish_target_detection(self, detections, timestamp):
        """Main detection handler - updates tracking, publishes confirmation and best target info."""
        if len(detections) == 0:
            self.pub_target_confirmation.publish(Bool(data=False))
            self.publish_target_list(timestamp)
            return

        # Update internal list with gating (also publishes ROIs and marker array)
        self.update_target_list(detections, timestamp)
        self.publish_target_list(timestamp)

        # Publish confirmation of detection
        self.pub_target_confirmation.publish(Bool(data=True))

        # Find best detection and publish its type
        best = max(detections, key=lambda d: d.confidence)
        label_str = labels[best.label] if best.label < len(labels) else str(best.label)
        
        type_msg = String()
        if label_str == "marker" and (self.marker_id is not None):
            type_msg.data = "marker_{}".format(str(self.marker_id))
        else:
            type_msg.data = label_str
        self.pub_target_type.publish(type_msg)

        rospy.loginfo("Best target: {} conf:{:.2f}".format(type_msg.data, best.confidence))
        rospy.loginfo("Total unique targets tracked: {}".format(len(self.detected_targets)))

    # -------------------- RViz Marker Publishing --------------------

    def _create_marker(self, x, y, z, marker_id, marker_ns, marker_type, text="", 
                      r=0.0, g=0.2, b=0.8, a=1.0, scale=0.3):
        """Unified marker creation - handles both CUBE and TEXT markers."""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = marker_ns
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.lifetime = rospy.Duration(0)  # Persistent
        m.frame_locked = True

        # Position
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z if marker_type == Marker.CUBE else z + 0.35  # Offset text above cube
        m.pose.orientation.w = 1.0

        # Scale and color
        if marker_type == Marker.CUBE:
            m.scale.x = scale
            m.scale.y = scale
            m.scale.z = scale * 0.3
            m.color.r = float(r)
            m.color.g = float(g)
            m.color.b = float(b)
            m.color.a = float(a)
        else:  # TEXT_VIEW_FACING
            m.scale.z = 0.2
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = float(a)
            m.text = text

        return m

    def publish_marker_array_snapshot(self, targets):
        """
        Publish a full MarkerArray of all currently tracked targets.
        Consolidates all marker creation and publishing in one place.
        Uses hash of marker_id_str for unique integer IDs.
        """
        arr = MarkerArray()
        for t in targets:
            # Use hash of the marker_id_str for a consistent integer ID
            marker_int_id = hash(t.marker_id_str) % (2**31)  # Keep within int32 range
            text_marker_int_id = (hash(t.marker_id_str + "_text") % (2**31))  # Different ID for text
            
            # Create cube marker
            cube = self._create_marker(
                t.world_x, t.world_y, t.world_z,
                marker_id=marker_int_id,
                marker_ns="detected_targets",
                marker_type=Marker.CUBE,
                r=0.1, g=0.6, b=0.2, a=0.9, scale=0.35
            )
            
            # Create text label marker
            label_text = "{} ({:.2f})".format(t.marker_id_str, t.confidence)
            text = self._create_marker(
                t.world_x, t.world_y, t.world_z,
                marker_id=text_marker_int_id,
                marker_ns="detected_targets_text",
                marker_type=Marker.TEXT_VIEW_FACING,
                text=label_text,
                a=1.0
            )
            
            arr.markers.append(cube)
            arr.markers.append(text)
        
        self.pub_marker_array.publish(arr)

    # -------------------- OAK-D / DepthAI loop --------------------

    def run(self):
        modelPathName = f'{modelsPath}/{modelName}/{modelName}.blob'
        rospy.loginfo(f"NN metadata: {metadata}")
        nnPath = str((Path(__file__).parent / Path(modelPathName)).resolve().absolute())
        rospy.loginfo(f"NN blob path: {nnPath}")

        pipeline = self.createPipeline(nnPath)

        with dai.Device() as device:
            cams = device.getConnectedCameras()
            if cam_source != "rgb":
                raise RuntimeError("This script currently requires the rgb camera. Connected: {}".format(cams))
            device.startPipeline(pipeline)

            q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            detections = []
            start_time = time.time()
            counter = 0
            fps = 0.0

            while not rospy.is_shutdown():
                found_classes = []
                inRgb = q_nn_input.get()
                inDet = q_nn.get()

                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                else:
                    rospy.logwarn("Empty camera frame; continuing...")
                    continue

                current_time = rospy.Time.now()

                if inDet is not None:
                    detections = inDet.detections
                    for d in detections:
                        label_str = labels[d.label] if d.label < len(labels) else str(d.label)
                        rospy.loginfo("{},{},{},{},{},{}".format(
                            label_str, d.confidence, d.xmin, d.ymin, d.xmax, d.ymax))
                        found_classes.append(label_str)
                    found_classes = np.unique(found_classes)
                    overlay = self.show_yolo(frame, detections)

                    # Publish detections to SPaR/Breadcrumb-compatible topics; this also publishes markers
                    self.publish_target_detection(detections, current_time)

                else:
                    rospy.logwarn("Detections empty; publishing no-confirmation.")
                    self.pub_target_confirmation.publish(Bool(data=False))
                    self.publish_target_list(current_time)
                    continue

                if frame is not None:
                    cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 4),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    cv2.putText(overlay, "Found: {}".format(list(found_classes)), (2, 14),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    cv2.putText(overlay, "Tracked: {}".format(len(self.detected_targets)), (2, 28),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))

                    self.publish_to_ros(frame)
                    self.publish_detect_to_ros(overlay)
                    # Also refresh camera info stamp
                    self.publish_camera_info()

                counter += 1
                if (time.time() - start_time) > 1.0:
                    fps = counter / (time.time() - start_time)
                    counter = 0
                    start_time = time.time()

    def publish_to_ros(self, frame):
        # Compressed
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.header.frame_id = "camera_frame"
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
        self.pub_image.publish(msg_out)

        # Raw
        msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
        msg_img_raw.header.stamp = rospy.Time.now()
        msg_img_raw.header.frame_id = "camera_frame"
        self.pub_image_raw.publish(msg_img_raw)

    def publish_detect_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.header.frame_id = "camera_frame"
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
        self.pub_image_detect.publish(msg_out)

    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def show_yolo(self, frame, detections):
        color = (255, 0, 0)
        overlay = frame.copy()
        for d in detections:
            bbox = self.frameNorm(overlay, (d.xmin, d.ymin, d.xmax, d.ymax))
            label_str = labels[d.label] if d.label < len(labels) else str(d.label)
            cv2.putText(overlay, label_str, (bbox[0] + 10, bbox[1] + 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(overlay, f"{int(d.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        return overlay

    def createPipeline(self, nnPath):
        pipeline = dai.Pipeline()
        # OpenVINO version may vary with your blob; adjust as needed
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
            rospy.loginfo("Using RGB camera...")
        elif cam_source == 'left':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
            rospy.loginfo("Using Mono LEFT camera...")
        elif cam_source == 'right':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            rospy.loginfo("Using Mono RIGHT camera...")

        if cam_source != 'rgb':
            manip = pipeline.create(dai.node.ImageManip)
            manip.setResize(self.nn_shape_w, self.nn_shape_h)
            manip.setKeepAspectRatio(True)
            manip.setFrameType(dai.RawImgFrame.Type.RGB888p)
            cam.out.link(manip.inputImage)
            manip.out.link(detection_nn.input)

        # Passthrough RGB frames used for overlay
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("nn_input")
        xout_rgb.input.setBlocking(False)
        detection_nn.passthrough.link(xout_rgb.input)

        # Detection output
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

    # Use run() loop until shutdown
    while not rospy.is_shutdown():
        dai_cam.run()

    dai_cam.shutdown()

if __name__ == '__main__':
    main()
