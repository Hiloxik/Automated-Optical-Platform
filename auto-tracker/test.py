import PySpin
import cv2
import numpy as np

def acquire_images(cam):
    """
    Acquires and displays images from a device.
    """
    try:
        cam.BeginAcquisition()
        print('Acquiring images...')
        while True:
            try:
                image_result = cam.GetNextImage()

                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                else:
                    # Assuming the camera is in a color format like RGB8 or BGR8
                    image_data = image_result.GetNDArray()

                    # Resize the image (you can adjust the size as needed)
                    resized_image = cv2.resize(image_data, (640, 480))  # Example size: 640x480

                    print(resized_image)

                    # Display the image using OpenCV
                    cv2.imshow('FLIR Camera Frame', resized_image)

                    if cv2.waitKey(1) == ord('q'):
                        break

                    image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)

    finally:
        cam.EndAcquisition()

def main():
    print(cv2.__version__)
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print('Not enough cameras!')
        return False

    cam = cam_list.GetByIndex(0)

    try:
        cam.Init()
        # Set pixel format to a color format if it's not already set
        # This is an example, you need to adapt it based on your camera's capabilities
        nodemap = cam.GetNodeMap()
        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
            node_pixel_format_bgr8 = node_pixel_format.GetEntryByName("BGR8")
            if PySpin.IsAvailable(node_pixel_format_bgr8) and PySpin.IsReadable(node_pixel_format_bgr8):
                pixel_format_bgr8 = node_pixel_format_bgr8.GetValue()
                node_pixel_format.SetIntValue(pixel_format_bgr8)

        acquire_images(cam)
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
    finally:
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
