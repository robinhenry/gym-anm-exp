import cv2
import argparse
import numpy as np
import pyautogui
import os
import time


def record(output_path, length, fps):

    if '.avi' not in output_path:
        output_path += '.avi'

    while os.path.exists(output_path):
        output_path = ''.join(output_path.split('.')[:-1]) + '1.' + output_path.split('.')[-1]
    print('Saving video to: ', output_path)

    img = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    height, width, channels = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start = time.time()

    while time.time() - start < length:
        try:
            img = pyautogui.screenshot()
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            out.write(image)
        except KeyboardInterrupt:
            break

    print("finished")
    out.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help="Path to output video file")
    parser.add_argument('--length', '-l', type=int, default=30, help='Length of recording (in sec)')
    parser.add_argument('--fps', type=float, default=10, help='Frame per second in saved video')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    record(args.output, args.length, args.fps)
