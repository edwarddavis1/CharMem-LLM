document.addEventListener("DOMContentLoaded", () => {
    const chatMessages = document.getElementById("chat-messages");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    let socket = null;
    let isConnected = false;

    // Connect to WebSocket
    function connectWebSocket() {
        // Create WebSocket connection
        // (The ws: protocol triggers a handshake with the server)
        // (If HTTPS, use wss: for secure WebSocket)
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

        // Connection opened
        socket.addEventListener("open", (event) => {
            isConnected = true;
            console.log("Connected to WebSocket server");
        });

        // Listen for messages
        socket.addEventListener("message", (event) => {
            const message = event.data;

            // Remove typing indicator if it exists
            const typingIndicator = document.querySelector(
                ".typing-indicator-container"
            );
            if (typingIndicator) {
                chatMessages.removeChild(typingIndicator);
            }

            // If the message is the "thinking" message, show typing indicator
            if (message === "Bot is thinking...") {
                appendTypingIndicator();
            } else {
                // Otherwise, display the actual message
                appendBotMessage(message);

                // Scroll to the bottom
                scrollToBottom();
            }
        });

        // Connection closed
        socket.addEventListener("close", (event) => {
            isConnected = false;
            console.log("Disconnected from WebSocket server");

            // Try to reconnect after a delay
            setTimeout(() => {
                connectWebSocket();
            }, 3000);
        });

        // Connection error
        socket.addEventListener("error", (error) => {
            console.error("WebSocket error:", error);
        });
    }

    // Connect to WebSocket
    connectWebSocket();

    // Send message function
    function sendMessage() {
        const message = userInput.value.trim();

        if (message && isConnected) {
            // Display user message
            appendUserMessage(message);

            // Clear input
            userInput.value = "";

            // Send message to server
            socket.send(message);

            // Scroll to the bottom
            scrollToBottom();
        }
    }

    // Append user message to chat
    function appendUserMessage(message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", "user");
        messageElement.innerHTML = `
            <div class="message-content">
                <p>${escapeHTML(message)}</p>
            </div>
        `;
        chatMessages.appendChild(messageElement);
    }

    // Append bot message to chat
    function appendBotMessage(message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", "bot");
        messageElement.innerHTML = `
            <div class="message-content">
                <p>${formatMessage(message)}</p>
            </div>
        `;
        chatMessages.appendChild(messageElement);
    }

    // Append typing indicator
    function appendTypingIndicator() {
        const indicatorElement = document.createElement("div");
        indicatorElement.classList.add(
            "message",
            "bot",
            "typing-indicator-container"
        );
        indicatorElement.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(indicatorElement);
        scrollToBottom();
    }

    // Format message (convert URLs to links, etc.)
    function formatMessage(message) {
        // Escape HTML
        let escapedMessage = escapeHTML(message);

        // Convert URLs to links
        escapedMessage = escapedMessage.replace(
            /(https?:\/\/[^\s]+)/g,
            '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
        );

        // Convert line breaks to <br>
        escapedMessage = escapedMessage.replace(/\n/g, "<br>");

        return escapedMessage;
    }

    // Escape HTML to prevent XSS
    function escapeHTML(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // Scroll to the bottom of the chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Event listeners
    sendBtn.addEventListener("click", sendMessage);

    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    // Focus input field
    userInput.focus();
});
