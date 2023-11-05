var A = 1;
var B = 1;

// Adjust these values to make the torus twice the size and centered
var R1 = 4; // Increase to make R1 twice as large
var R2 = 8; // Increase to make R2 twice as large
var K1 = 600; // Increase to scale K1
var K2 = 20; // Increase to scale K2

var canvasframe = function() {
var canvastag = document.getElementById('canvasdonut');
var ctx = canvastag.getContext('2d');
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

A += 0.07;
B += 0.03;

var cA = Math.cos(A);
var sA = Math.sin(A);
var cB = Math.cos(B);
var sB = Math.sin(B);

var centerX = ctx.canvas.width / 2;
var centerY = ctx.canvas.height / 2;

for (var j = 0; j < 6.28; j += 0.1) { // Larger step for fewer points
    var ct = Math.cos(j);
    var st = Math.sin(j);

    for (var i = 0; i < 6.28; i += 0.1) {   // Larger step for fewer points
    var sp = Math.sin(i);
    var cp = Math.cos(i);
    var ox = R2 + R1 * ct;
    var oy = R1 * st;

    var x = ox * (cB * cp + sA * sB * sp) - oy * cA * sB;
    var y = ox * (sB * cp - sA * cB * sp) + oy * cA * cB;
    var ooz = 1 / (K2 + cA * ox * sp + sA * oy);
    var xp = centerX + K1 * ooz * x; // Center the torus
    var yp = centerY - K1 * ooz * y; // Center the torus
    var L = 0.7 * (cp * ct * sB - cA * ct * sp - sA * st + cB * (cA * st - ct * sA * sp));

    if (L > 0) {
        // Larger and more visible points
        ctx.fillStyle = 'rgba(255,255,255,' + (L + 0.3) + ')'; // Adjust opacity
        ctx.fillRect(xp, yp, 3, 3); // Increase the size of the points
    }
    }
}
};

var tmr2 = setInterval(canvasframe, 35); // Reduced interval for a faster animation

canvasframe();