function sendMessage() {
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");

    // Get the user's message
    const userMessage = userInput.value.trim();

    if (userMessage === "") {
        return;
    }

    // Display the user's message
    const userMessageElement = document.createElement("div");
    userMessageElement.classList.add("user-message");
    userMessageElement.textContent = "You: " + userMessage;
    chatBox.appendChild(userMessageElement);

    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;

    // Clear the input field
    userInput.value = "";

    // Respond to the user's message
    setTimeout(() => {
        const botMessageElement = document.createElement("div");
        botMessageElement.classList.add("bot-message");

        // Simple responses
        let botResponse = getBotResponse(userMessage);
        botMessageElement.textContent = "Bot: " + botResponse;
        chatBox.appendChild(botMessageElement);

        // Scroll to the bottom of the chat box
        chatBox.scrollTop = chatBox.scrollHeight;
    }, 500);
}

function getBotResponse(input) {
    // Basic responses for certain keywords
    input = input.toLowerCase();

    if (input.includes("hello")) {
        return "Hello! How can I help you today?";
    } else if (input.includes("how are you")) {
        return "I'm just a bot, but I'm here to assist you!";
    } else if (input.includes("bye")) {
        return "Goodbye! Have a great day!";
    } else {
        return "I'm not sure how to respond to that. Can you rephrase?";
    }
}