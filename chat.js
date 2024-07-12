function sendMessage() {
    var userInput = document.getElementById("userInput").value;
    var chatlog = document.getElementById("chatlog");

    // Display user message
    var userMessage = document.createElement("div");
    userMessage.textContent = "You: " + userInput;
    chatlog.appendChild(userMessage);

    // Send user input to the backend
    fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        // Display chatbot response
        var botMessage = document.createElement("div");
        botMessage.textContent = "Bot: " + data.response;
        chatlog.appendChild(botMessage);
    })
    .catch(error => {
        console.error('Error:', error);
    });

    // Clear input
    document.getElementById("userInput").value = "";
}
