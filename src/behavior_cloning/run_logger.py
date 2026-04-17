import rclpy
from freedrive_logger import FreeDriveLogger

def main():
    rclpy.init()
    node = FreeDriveLogger()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        # Save data safely
        node.save()
        node.destroy_node()

        # Shutdown safely (avoid double shutdown)
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
