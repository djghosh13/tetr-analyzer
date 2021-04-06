var maxFrames = parseInt("[MAX_FRAMES]");
var side = parseInt("[PLAYER_SIDE]");
var tag = "[DATA_TAG]";

const frameOffsetX = [187, 474][side], frameOffsetY = 84;
const frameWidth = 133, frameHeight = 295;

var imageData = new Uint8Array(4 * frameWidth * frameHeight);
var captureFrames = [];

function hsl(R, G, B) {
    let M = Math.max(R, G, B);
    let m = Math.min(R, G, B);
    let C = M - m;
    // Calculate hue
    let H = 0;
    if (C == 0) {
        H = 0;
    } else if (M == R) {
        H = ((G - B) / C + 6) % 6;
    } else if (M == G) {
        H = (B - R) / C + 2;
    } else {
        H = (R - G) / C + 4;
    }
    H *= 60;
    // Calculate value
    let V = M / 255 * 100;
    // Calculate saturation
    let S = (V == 0) ? 0 : C / M * 100;
    return [H, S, V];
}

const pieceColors = {
    "S": [[82, 70, 80], [0.7, 0.2, 0.1]],
    "Z": [[356, 70, 80], [0.7, 0.2, 0.1]],
    "T": [[306, 70, 80], [0.7, 0.2, 0.1]],
    "I": [[158, 70, 80], [0.7, 0.2, 0.1]],
    "L": [[22, 70, 80], [0.7, 0.2, 0.1]],
    "J": [[250, 70, 80], [0.7, 0.2, 0.1]],
    "O": [[47, 70, 80], [0.7, 0.2, 0.1]],
    "G": [[0, 0, 29], [0, 0.4, 0.6]],
    "-": [[0, 0, 0], [0, 0.4, 0.6]]
};

function nearestNeighbor(color) {
    color = hsl(...color);
    let bestPiece = null, minDist = 1e6;
    for (const [piece, [pColor, pWeight]] of Object.entries(pieceColors)) {
        let dist = Math.min((color[0] - pColor[0] + 360) % 360, (pColor[0] - color[0] + 360) % 360) * pWeight[0]
            + Math.abs(color[1] - pColor[1]) * pWeight[1]
            + Math.abs(color[2] - pColor[2]) * pWeight[2];
        if (dist < minDist) {
            bestPiece = piece, minDist = dist;
        }
    }
    return bestPiece;
}

function readGrid() {
    let grid = [];
    for (let cy = 0; cy < 22; cy++) {
        let pieces = [];
        let yp = Math.floor(cy * 13.40 + 9);
        for (let cx = 0; cx < 10; cx++) {
            let xp = Math.floor(cx * 13.40 + 5);
            if (cy == 16 && (cx == 6 || cx == 7)) {
                yp -= 2;
                xp -= 3;
            }
            let idx = (yp * frameWidth + xp) * 4;
            let color = [imageData[idx + 0], imageData[idx + 1], imageData[idx + 2]];
            let piece = nearestNeighbor(color);
            pieces.push(piece);
        }
        grid.unshift(pieces);
    }
    return grid;
}

function capture() {
    let frame = -1;
    try {
        frame = parseInt(document.getElementById("replaytools_timestamp").innerText.split("frame ")[1]);
        if (frame >= maxFrames) throw new Error();
    } catch (error) {
        console.log("Captured " + captureFrames.length + " frames of grid");
        let storage = document.createElement("div");
        storage.setAttribute("id", "captured-" + tag);
        storage.style.display = "none";
        storage.innerText = JSON.stringify(captureFrames);
        document.body.appendChild(storage);
        return;
    }
    let canvas = document.getElementById("pixi");
    let gl = canvas.getContext("webgl2") || gl.canvas.getContext("webgl");
    // gl.enable(gl.SCISSOR_TEST);
    // gl.scissor(frameOffsetX, frameOffsetY, frameWidth, frameHeight);
    gl.readPixels(frameOffsetX, frameOffsetY, frameWidth, frameHeight, gl.RGBA, gl.UNSIGNED_BYTE, imageData);
    let grid = readGrid();
    // Save captured frame
    if (captureFrames.length && captureFrames[captureFrames.length - 1]["frame"] == frame) {
        captureFrames.pop();
    }
    captureFrames.push({
        "frame": frame,
        "value": grid
    });
    // Re-run
    requestAnimationFrame(capture);
}

for (let element of document.querySelectorAll("#captured-" + tag)) {
    element.remove();
}
for (let element of document.querySelectorAll("#custom-style")) {
    element.remove();
}
{
    let element = document.createElement("style");
    element.setAttribute("id", "custom-style");
    element.setAttribute("type", "text/css");
    element.innerText = `
        .watch_header .keystone {
            background-color: #266dcd !important;
        }
        .watch_header .data>span {
            color: #266dcd !important;
        }
    `;
    document.body.appendChild(element);
}
var captureID = requestAnimationFrame(capture);

return captureID;