// Drop file script
let target = arguments[0];
let document = target.ownerDocument || document;
let window = document.defaultView || window;

let input = document.createElement("input");
input.type = "file";
// input.addEventListener("change", function () {
//     let rect = target.getBoundingClientRect();
//     let x = rect.left + (rect.width >> 1);
//     let y = rect.top + (rect.height >> 1);
//     // Create data transfer
//     let dataTransfer = new DataTransfer();
//     dataTransfer.dropEffect = "copy";
//     dataTransfer.items.add(this.files[0]);

//     ["dragenter", "dragover", "drop"].forEach(name => {
//         let event = new DragEvent(name, {
//             "bubbles": true,
//             "dataTransfer": dataTransfer
//         });
//         target.dispatchEvent(event);
//         console.log(event);
//     });

//     setTimeout(() => document.body.removeChild(input), 25);
// });

document.body.appendChild(input);
return input;