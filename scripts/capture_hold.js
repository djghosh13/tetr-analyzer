var maxFrames = parseInt("[MAX_FRAMES]");
var side = parseInt("[PLAYER_SIDE]");
var tag = "[DATA_TAG]";

const frameOffsetX = [110, 397, 243][side], frameOffsetY = 300;
const frameWidth = 68, frameHeight = 41;

var imageData = new Uint8Array(4 * frameWidth * frameHeight);
var captureFrames = [];


const keyCoords = [
    8 + 19 * frameWidth,
    16 + 12 * frameWidth,
    33 + 12 * frameWidth,
    51 + 12 * frameWidth,
    16 + 26 * frameWidth,
    33 + 26 * frameWidth,
    51 + 26 * frameWidth
];

function notBlack(illum) {
    return (illum > 10) + 0;
}

function readPiece(imageData, width, height) {
    var key = 0;
    for (let idx of keyCoords) {
        // Determine illumination
        let illum = imageData[idx * 4 + 0] + imageData[idx * 4 + 1] + imageData[idx * 4 + 2];
        key = (key << 1) | notBlack(illum);
    }
    switch (key) {
        case 0:
            return "-";
        case 0b0010010:
            return "O";
        case 0b0011110:
            return "Z";
        case 0b0110011:
            return "S";
        case 0b0111001:
            return "L";
        case 0b0111010:
            return "T";
        case 0b0111100:
            return "J";
        case 0b1111000:
            return "I";
        default:
            return "unk";
    }
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