var target = arguments[0],
    offsetX = arguments[1],
    offsetY = arguments[2],
    document = target.ownerDocument || document,
    window = document.defaultView || window;

var input = document.createElement('input');
input.type = 'file';
input.onchange = function () {
    var rect = target.getBoundingClientRect(),
        x = rect.left + (offsetX || (rect.width >> 1)),
        y = rect.top + (offsetY || (rect.height >> 1)),
        dataTransfer = { files: this.files };

    ['dragenter', 'dragover', 'drop'].forEach(function (name) {
        var evt = document.createEvent('MouseEvent');
        evt.initMouseEvent(name, !0, !0, window, 0, 0, 0, x, y, !1, !1, !1, !1, 0, null);
        evt.dataTransfer = dataTransfer;
        target.dispatchEvent(evt);
    });

    setTimeout(function () { document.body.removeChild(input); }, 25);
};
document.body.appendChild(input);
return input;