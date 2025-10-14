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
confidenceThreshold = 0.8
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", [])

class DetectedTarget:
    """Store detected target information"""
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
            self.nn_shape_w, self.nn_shape_h = 416, 416

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
        # self.sub_uav_pose = rospy.Subscriber('/uavasr/pose', PoseStamped, self.callback_uav_pose)
        self.sub_pose = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.callback_uav_pose)
        self.current_uav_pose = None

        # Tracking
        self.detected_targets = []
        self.detected_gating_set = set()  # (label, marker_id) gating
        self.target_id_counter = 0
        self.target_timeout = 2.5  # seconds

        # Camera extrinsics (relative to UAV base)
        self.camera_offset_x = 0.12
        self.camera_offset_y = 0.0
        self.camera_offset_z = -0.1

        # Intrinsics (approx)
        self.fx, self.fy = 615.381, 615.381
        self.cx, self.cy = 320.0, 240.0

        # Publish CameraInfo periodically (and we also send on new frames)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_camera_info, oneshot=False)

        # Marker array snapshot
        self.marker_array = MarkerArray()
        self.marker_seq = 0  # Unique marker ID source

        rospy.loginfo("Publishing images to: {}".format(self.pub_topic))
        rospy.loginfo("Target topics: confirmation={}, type={}, roi={}, list={}".format(
            self.pub_topic_target_confirmation, self.pub_topic_target_type,
            self.pub_topic_target_roi, self.pub_topic_target_list))
        rospy.loginfo("Marker topics: marker={}, array={}".format(self.pub_topic_marker, self.pub_topic_marker_array))

        self.br = CvBridge()
        rospy.on_shutdown(self.shutdown)

    def callback_uav_pose(self, msg):
        self.current_uav_pose = msg

    def aruco_callback(self, msg):
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

        # Remove stale
        self.detected_targets = [
            t for t in self.detected_targets
            if (current_time_sec - t.last_seen.to_sec()) < self.target_timeout
        ]

        for detection in new_detections:
            # Gate by (label, marker_id) to reduce duplicates
            gate_key = (labels[detection.label] if detection.label < len(labels) else str(detection.label),
                        self.marker_id if self.marker_id is not None else "none")
            if gate_key in self.detected_gating_set:
                rospy.loginfo(f"[GATE] Duplicate ignored: {gate_key}")
                continue
            self.detected_gating_set.add(gate_key)
            rospy.loginfo(f"[GATE] New accepted: {gate_key}")

            # Center & area
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
            if world_z < 0:
                rospy.loginfo("Detection skipped: below ground")
                continue

            label_str = labels[detection.label] if detection.label < len(labels) else str(detection.label)
            new_target = DetectedTarget(
                self.target_id_counter, label_str, detection.confidence,
                world_x, world_y, world_z, timestamp
            )
            self.detected_targets.append(new_target)
            self.target_id_counter += 1

            rospy.loginfo("New target ID={} {} @ [{:.2f}, {:.2f}, {:.2f}] conf:{:.2f}".format(
                new_target.target_id, new_target.label, world_x, world_y, world_z, detection.confidence))

            # Publish per-target ROI immediately
            roi_msg = PoseStamped()
            roi_msg.header.stamp = timestamp
            roi_msg.header.frame_id = "map"
            roi_msg.pose.position.x = world_x
            roi_msg.pose.position.y = world_y
            roi_msg.pose.position.z = world_z
            roi_msg.pose.orientation.w = 1.0
            self.pub_target_roi.publish(roi_msg)

            # Also publish a marker for this target (dynamic; latch keeps the last one if stream pauses)
            self.publish_marker(world_x, world_y, world_z,
                                marker_id=self.marker_id,
                                label=new_target.label,
                                target_id=new_target.target_id)

        # After processing, refresh a full MarkerArray snapshot of all tracked targets
        self.publish_marker_array_snapshot(self.detected_targets)

    def publish_target_list(self, timestamp):
        if len(self.detected_targets) == 0:
            msg = String(data="No targets detected")
            self.pub_target_list.publish(msg)
            return

        items = []
        for t in self.detected_targets:
            items.append("ID:{} Type:{} Pos:[{:.2f},{:.2f},{:.2f}] Conf:{:.2f}".format(
                t.target_id, t.label, t.world_x, t.world_y, t.world_z, t.confidence))
        self.pub_target_list.publish(String(data="Targets: " + " | ".join(items)))

    def publish_target_detection(self, detections, timestamp):
        if len(detections) == 0:
            self.pub_target_confirmation.publish(Bool(data=False))
            self.publish_target_list(timestamp)
            return

        # Update internal list (& per-target ROI + markers)
        self.update_target_list(detections, timestamp)
        self.publish_target_list(timestamp)

        # Best detection details for quick consumers
        best = max(detections, key=lambda d: d.confidence)

        cx = (best.xmin + best.xmax) / 2.0 * self.nn_shape_w
        cy = (best.ymin + best.ymax) / 2.0 * self.nn_shape_h
        bw = (best.xmax - best.xmin) * self.nn_shape_w
        bh = (best.ymax - best.ymin) * self.nn_shape_h
        area = bw * bh
        max_area = self.nn_shape_w * self.nn_shape_h
        depth_estimate = max(0.5, 5.0 * (1.0 - area / max_area))

        world_coords = self.pixel_to_world_coordinates(cx, cy, depth_estimate)
        if world_coords is None:
            rospy.logwarn("Cannot localize target - UAV pose unavailable")
            return

        world_x, world_y, world_z = world_coords

        self.pub_target_confirmation.publish(Bool(data=True))

        label_str = labels[best.label] if best.label < len(labels) else str(best.label)
        typemsg = String()
        if label_str == "marker" and (self.marker_id is not None):
            typemsg.data = "marker_{}".format(str(self.marker_id))
        else:
            typemsg.data = label_str
        self.pub_target_type.publish(typemsg)

        roi_msg = PoseStamped()
        roi_msg.header.stamp = timestamp
        roi_msg.header.frame_id = "map"
        roi_msg.pose.position.x = world_x
        roi_msg.pose.position.y = world_y
        roi_msg.pose.position.z = world_z
        roi_msg.pose.orientation.w = 1.0
        self.pub_target_roi.publish(roi_msg)

        # Quick marker for the best detection too
        self.publish_marker(world_x, world_y, world_z, marker_id=self.marker_id,
                            label=typemsg.data, target_id=None)

        rospy.loginfo("Best target: {} @ [{:.2f}, {:.2f}, {:.2f}] conf:{:.2f}".format(
            typemsg.data, world_x, world_y, world_z, best.confidence))
        rospy.loginfo("Total unique targets tracked: {}".format(len(self.detected_targets)))

    # -------------------- RViz Marker Publishing --------------------

    def _make_marker(self, x, y, z, marker_id_int, label_text, r=0.0, g=0.2, b=0.8, a=1.0, scale=0.3):
        """Create a single CUBE marker at (x,y,z) with text label displayed via namespace/id."""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "detected_targets"
        m.id = marker_id_int
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.lifetime = rospy.Duration(0)   # infinite
        m.frame_locked = True

        # Pose
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # Scale (meters)
        m.scale.x = scale
        m.scale.y = scale
        m.scale.z = scale * 0.3

        # Color
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = float(a)

        return m

    def _make_text_marker(self, x, y, z, marker_id_int, text, a=1.0):
        """Optional: Add a floating TEXT_VIEW_FACING marker above the cube."""
        t = Marker()
        t.header.frame_id = "map"
        t.header.stamp = rospy.Time.now()
        t.ns = "detected_targets_text"
        t.id = marker_id_int
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.lifetime = rospy.Duration(0)
        t.frame_locked = True

        t.pose.position.x = x
        t.pose.position.y = y
        t.pose.position.z = z + 0.35
        t.pose.orientation.w = 1.0

        t.scale.z = 0.2  # text height (meters)
        t.color.r = 1.0
        t.color.g = 1.0
        t.color.b = 1.0
        t.color.a = a
        t.text = text
        return t

    def publish_marker(self, world_x, world_y, world_z, marker_id, label, target_id=None,
                       r=0.0, g=0.2, b=0.8, a=1.0):
        """
        Publish a single marker (CUBE) and a label (TEXT) at the detected position.
        Uses latch=True so RViz holds last marker when stream pauses.
        """
        # Make a stable-ish integer id: prefer target_id if provided, else rolling seq
        if target_id is not None:
            marker_id_int = int(target_id)
        else:
            self.marker_seq += 1
            marker_id_int = self.marker_seq

        m = self._make_marker(world_x, world_y, world_z, marker_id_int, label, r, g, b, a)
        t = self._make_text_marker(world_x, world_y, world_z, marker_id_int + 100000,  # offset id space
                                   text=(f"{label}" + (f" [{marker_id}]" if marker_id else "")))

        self.pub_marker.publish(m)
        self.pub_marker.publish(t)  # publish text as well (same latched topic is fine)

    def publish_marker_array_snapshot(self, targets):
        """
        Publish a full snapshot of all currently tracked targets as a MarkerArray.
        Useful for RViz to see the whole set at once. Latched.
        """
        arr = MarkerArray()
        for t in targets:
            cube = self._make_marker(t.world_x, t.world_y, t.world_z, t.target_id, t.label,
                                     r=0.1, g=0.6, b=0.2, a=0.9, scale=0.35)
            text = self._make_text_marker(t.world_x, t.world_y, t.world_z, t.target_id + 200000,
                                          text=f"ID:{t.target_id} {t.label} ({t.confidence:.2f})")
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
