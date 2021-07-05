import numpy as np
import cv2
from solver_tools import to_array, piece_filters


GRID_HEIGHT, GRID_WIDTH = 22, 10
VERBOSE = False
VIDEO_SCALE = 1

def _log(*args, **kwargs):
    if VERBOSE:
        print("[Renderer] ", *args, **kwargs)

class RendererConfig:
    @staticmethod
    def set_verbose(verbose=True):
        global VERBOSE
        VERBOSE = verbose
        return RendererConfig

    @staticmethod
    def set_scale(scale):
        global VIDEO_SCALE
        VIDEO_SCALE = np.clip(int(scale), 1, None)
        return RendererConfig


piece_colors = {
    "S": [0, 240, 0],
    "Z": [239, 1, 0],
    "T": [159, 0, 243],
    "I": [1, 239, 241],
    "L": [239, 159, 0],
    "J": [1, 0, 239],
    "O": [239, 240, 0],
    "G": [120, 120, 120],
    "-": [0, 0, 0],
    "*": [255, 255, 255]
}

def create_image_grid(grid, out=None):
    H, W = len(grid), len(grid[0])
    if out is None:
        out = np.zeros((H, W, 3), dtype=np.uint8)
    assert out.shape == (H, W, 3)
    # Draw grid
    for i in range(H):
        for j in range(W):
            out[i, j] = piece_colors[grid[i][j]][::-1]
    return out


class ReplayVideo:
    def __init__(self):
        self.frame = 0
        self.video = []
        self.reset()

    def reset(self):
        self.frame = 0
        self.video.append(np.zeros([GRID_HEIGHT, GRID_WIDTH], dtype=np.uint8))

    def extend_by(self, n_frames):
        self.video += [self.video[-1]] * n_frames

    def extend_to(self, frame):
        self.video += [self.video[-1]] * (frame - self.frame)

    def render(self, *events):
        for ev in events:
            if ev["type"] == "garbage": continue
            action = ev["event"]
            self.extend_to(ev["frame"] - 1)
            # Draw new frame
            grid = to_array(ev["board"])
            ftr = piece_filters[action.piece][action.rotation]
            grid[action.y:action.y + ftr.shape[0], action.x:action.x + ftr.shape[1]][ftr] = "*"
            self.video.append(create_image_grid(grid))
            self.frame = ev["frame"]

    def save(self, filename, fps=60):
        size = (GRID_WIDTH * VIDEO_SCALE, GRID_HEIGHT * VIDEO_SCALE)
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for frame in self.video:
            out.write(cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST))
        out.release()
        _log(f"Saved {len(self.video)} frame(s) to '{filename}'")


def display_image(image, title="Replay"):
    cv2.imshow(title, cv2.resize(image, standard_size[image.shape[:2]], interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_video(video):
    for frame in video:
        cv2.imshow("Replay", cv2.resize(frame, standard_size[frame.shape[:2]], interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(13) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def save_video(filename, video):
    size = standard_size[video.shape[1:3]]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 120, size)
    for frame in video:
        out.write(cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST))
    out.release()