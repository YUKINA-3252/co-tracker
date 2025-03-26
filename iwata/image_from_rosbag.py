import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bag_path = "rosbag/hiro_demo_2025-01-03-16-23-41.bag"
image_topic = "/head_camera/rgb/image_raw/compressed"
depth_topic = "/head_camera/depth/image_rect_raw"

# CvBridgeの初期化
bridge = CvBridge()

output_image_path = "rgb_image/output_image.jpg"
output_depth_path = "depth_image/output_depth.png"

def save_image_after_start_time(bag_path, image_topic, save_path, start_delay=3.0):
    with rosbag.Bag(bag_path, 'r') as bag:
        start_time = None

        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            if start_time is None:
                start_time = t.to_sec()

            elapsed_time = t.to_sec() - start_time

            if elapsed_time >= start_delay:
                try:
                    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                    cv2.imwrite(save_path, cv_image)
                    print(f"Image saved to {save_path} at {elapsed_time:.2f} seconds.")
                except Exception as e:
                    print(f"Failed to save image: {e}")
                break


def save_depth_after_start_time(bag_path, image_topic, save_path, start_delay=3.0):
    with rosbag.Bag(bag_path, 'r') as bag:
        start_time = None

        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            if start_time is None:
                start_time = t.to_sec()

            elapsed_time = t.to_sec() - start_time

            if elapsed_time >= start_delay:
                try:
                    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                    cv2.imwrite(save_path, depth_image)
                    print(f"Image saved to {save_path} at {elapsed_time:.2f} seconds.")
                except Exception as e:
                    print(f"Failed to save image: {e}")
                break


if __name__ == "__main__":
    save_image_after_start_time(bag_path, image_topic, output_image_path, start_delay=300.0)
    save_depth_after_start_time(bag_path, depth_topic, output_depth_path, start_delay=300.0)
