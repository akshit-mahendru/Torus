function updateTime() {
    const timeElement = document.getElementById('time');
    const currentTime = new Date();
    let hours = currentTime.getHours();
    const minutes = currentTime.getMinutes().toString().padStart(2, '0');
    const seconds = currentTime.getSeconds().toString().padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';

    // Convert to 12-hour format
    hours = hours % 12 || 12;

    const formattedTime = `${hours}:${minutes}:${seconds} ${ampm}`;
    timeElement.textContent = formattedTime;
}

// Call the updateTime function to set the initial time
updateTime();

// Update the time every second (1000 milliseconds)
setInterval(updateTime, 1000);
