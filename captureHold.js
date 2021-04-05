var maxFrames = parseInt("[MAX_FRAMES]");
var side = parseInt("[PLAYER_SIDE]");
var tag = "[DATA_TAG]";

const frameOffsetX = [143, 430][side], frameOffsetY = 318;
const frameWidth = 2, frameHeight = 1;

var imageData = new Uint8Array(4 * frameWidth * frameHeight);
var captureFrames = [];

const pieceColors = {
    "S": [152, 196, 78],
    "Z": [199, 78, 85],
    "T": [181, 89, 171],
    "I": [76, 199, 153],
    "L": [197, 123, 78],
    "J": [105, 89, 181],
    "O": [206, 182, 84],
    "used": [50, 50, 50],
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

function readPiece(imageData, width, height) {
    // Determine average color
    let color = [0, 0, 0];
    let pixels = width * height;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let idx = (y * width + x) * 4;
            color[0] += imageData[idx + 0];
            color[1] += imageData[idx + 1];
            color[2] += imageData[idx + 2];
        }
    }
    color[0] /= pixels;
    color[1] /= pixels;
    color[2] /= pixels;
    return nearestNeighbor(color);
}

function capture() {
    let frame = -1;
    try {
        frame = parseInt(document.getElementById("replaytools_timestamp").innerText.split("frame ")[1]);
        if (frame >= maxFrames) throw new Error();
    } catch (error) {
        console.log("Captured " + captureFrames.length + " frames of hold piece");
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
    let holdpiece = readPiece(imageData, frameWidth, frameHeight);
    // Save captured frame
    if (captureFrames.length && captureFrames[captureFrames.length - 1]["frame"] == frame) {
        captureFrames.pop();
    }
    captureFrames.push({
        "frame": frame,
        "value": holdpiece
    });
    // Re-run
    requestAnimationFrame(capture);
}

for (let element of document.querySelectorAll("#captured-" + tag)) {
    element.remove();
}
var captureID = requestAnimationFrame(capture);

return captureID;