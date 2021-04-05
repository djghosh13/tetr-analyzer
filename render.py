import numpy as np
import cv2

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
piece_shapes = {
    "S": [[0, 1, 1, 0], [1, 1, 0, 0]],
    "Z": [[1, 1, 0, 0], [0, 1, 1, 0]],
    "T": [[0, 1, 0, 0], [1, 1, 1, 0]],
    "I": [[0, 0, 0, 0], [1, 1, 1, 1]],
    "L": [[0, 0, 1, 0], [1, 1, 1, 0]],
    "J": [[1, 0, 0, 0], [1, 1, 1, 0]],
    "O": [[0, 1, 1, 0], [0, 1, 1, 0]],
    "-": [[0, 0, 0, 0], [0, 0, 0, 0]]
}
standard_size = {
    (22, 18): (18 * 16, 22 * 16),
    (22, 18*2 + 4): ((18*2 + 4) * 16, 22 * 16)
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

def create_image_next(pieces, out=None):
    if out is None:
        out = np.zeros((20, 4, 3), dtype=np.uint8)
    assert out.shape == (20, 4, 3)
    # Draw next queue
    for i, p in enumerate(pieces):
        out[4*i + 1:4*i + 3] = np.reshape(piece_shapes[p], (2, 4, 1)) * piece_colors[p][::-1]
    return out

def create_slideshow(data, pieces, interval=30, interpolation=False):
    data = [{**f, "frame": i * interval} for i, f in enumerate(data)]
    return create_video(data, pieces, interpolation=interpolation)

def create_video(data, pieces, interpolation=True, length=None):
    N, H, W = data[-1]["frame"], len(data[0]["grid"]), len(data[0]["grid"][0])
    if length is None:
        length = N
    video = np.zeros((max(N, length) + 1, H, W + 8, 3), dtype=np.uint8)
    # Render main grid
    gridregion = video[:, :, 1:W + 1]
    lastframe = 0
    interpolated, maxdrop = 0, 0
    for d in data:
        frame, grid = d["frame"], d["grid"]
        create_image_grid(grid, out=gridregion[frame])
        for f in range(lastframe + 1, frame):
            t = (f - lastframe) / (frame - lastframe) if interpolation else 0
            gridregion[f] = (1 - t) * gridregion[lastframe] + t * gridregion[frame]
            interpolated += 1
        maxdrop = max(maxdrop, frame - lastframe - 1)
        lastframe = frame
    # print(f"Interpolated {interpolated} / {N} frames")
    # print(f"Dropped at most {maxdrop} frames")
    video[:, :2] //= 3
    # Divider
    video[:, :, [0, W + 1]] = 48
    # Render next pieces
    for i in range(N + 1):
        create_image_next(pieces[i], out=video[i, 1:-1, W + 3:W + 7])
    # Adjust to length
    if length < N:
        video = video[:length + 1]
    if length > N:
        video[N:] = video[N]
    return video

def side_by_side(videos):
    video = videos[0]
    bar = np.zeros(video.shape[:2] + (4, 3), dtype=video.dtype)
    for v in videos[1:]:
        assert v.shape[:2] == video.shape[:2]
        video = np.concatenate([video, bar, v], axis=2)
    return video

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