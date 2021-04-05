var maxFrames = parseInt("[MAX_FRAMES]");
var side = parseInt("[PLAYER_SIDE]");
var tag = "[DATA_TAG]";

const frameOffsetX = [323, 610][side], frameOffsetY = 139;
const frameWidth = 66, frameHeight = 201;

var imageData = new Uint8Array(4 * frameWidth * frameHeight);
var captureFrames = [];

const pieceColors = {
    "S": [139, 185, 63],
    "Z": [192, 70, 77],
    "T": [169, 76, 160],
    "I": [66, 190, 144],
    "L": [187, 113, 67],
    "J": [95, 79, 172],
    "O": [186, 162, 65],
    "-": [0, 0, 0]
};

function nearestNeighbor(color) {
    let bestPiece = null, minDist = 1e6;
    for (const [piece, pColor] of Object.entries(pieceColors)) {
        let dist = Math.abs(color[0] - pColor[0]) + Math.abs(color[1] - pColor[1]) + Math.abs(color[2] - pColor[2]);
        if (dist < minDist) {
            bestPiece = piece, minDist = dist;
        }
    }
    return bestPiece;
}

function readPieces(imageData, width, height) {
    let pieces = [];
    for (let p = 0; p < 5; p++) {
        // Determine average color
        let color = [0, 0, 0], pixels = 1;
        for (let y = Math.floor(p * height / 5); y < (p + 1) * height / 5; y++) {
            for (let x = 0; x < width; x++) {
                let idx = (y * width + x) * 4;
                if (imageData[idx + 0] + imageData[idx + 1] + imageData[idx + 2] > 30) {
                    color[0] += imageData[idx + 0];
                    color[1] += imageData[idx + 1];
                    color[2] += imageData[idx + 2];
                    pixels++;
                }
            }
        }
        color[0] /= pixels;
        color[1] /= pixels;
        color[2] /= pixels;
        const bestPiece = nearestNeighbor(color);
        pieces.unshift(bestPiece);
    }
    return pieces;
}

function capture() {
    let frame = -1;
    try {
        frame = parseInt(document.getElementById("replaytools_timestamp").innerText.split("frame ")[1]);
        if (frame >= maxFrames) throw new Error();
    } catch (error) {
        console.log("Captured " + captureFrames.length + " frames of next queue");
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
    let pieces = readPieces(imageData, frameWidth, frameHeight).reduce((a, x) => a + x, "");
    // Save captured frame
    if (captureFrames.length && captureFrames[captureFrames.length - 1]["frame"] == frame) {
        captureFrames.pop();
    }
    captureFrames.push({
        "frame": frame,
        "value": pieces
    });
    // Re-run
    requestAnimationFrame(capture);
}

for (let element of document.querySelectorAll("#captured-" + tag)) {
    element.remove();
}
var captureID = requestAnimationFrame(capture);

return captureID;