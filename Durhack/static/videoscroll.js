document.addEventListener("DOMContentLoaded", function() {
    console.log('Running');
    let vid = document.getElementById("vid");
    vid.muted = true; // Mute the video

    // Event listener for when the video ends
    vid.addEventListener("ended", function() {
        vid.currentTime = vid.duration; // Set currentTime to the end of the video
        vid.pause(); // Pause the video
    });

    let observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                vid.play();
            }
        });
    });
    observer.observe(vid);
});