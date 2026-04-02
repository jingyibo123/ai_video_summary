import os
import cv2
import numpy as np

def generate_video(output_path):
    fps = 30
    duration = 4
    width, height = 640, 480

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for t in range(duration * fps):
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 0-1s: 第一段幻灯片开头
        if t < fps:
            cv2.putText(img, "Slide 1: Video pipeline", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # 1-2s: 第一段幻灯片展示动画和光标
        elif t < 2 * fps:
            cv2.putText(img, "Slide 1: Video pipeline", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, "-> Fast Extract", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(img, (100 + t%50, 350), 10, (0,0,255), -1) # Mock laser pointer
        # 2-4s: 切到全新幻灯片
        else:
            img = np.zeros((height, width, 3), dtype=np.uint8) # Dark background
            cv2.putText(img, "Slide 2: VLM Merging", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(img)

    out.release()
    print(f"Mock video created at {output_path}")

if __name__ == "__main__":
    generate_video("sample.mp4")
