/**
 * CharMem Chat Application
 * Handles WebSocket communication, message display, and PDF uploads
 */

document.addEventListener("DOMContentLoaded", () => {
    // ========================================
    // DOM ELEMENTS
    // ========================================
    const chatMessages = document.getElementById("chat-messages");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const uploadBtn = document.getElementById("upload-btn");
    const pdfUpload = document.getElementById("pdf-upload");

    // ========================================
    // GLOBAL STATE
    // ========================================
    let socket = null;
    let isConnected = false;

    // ========================================
    // WEBSOCKET FUNCTIONS
    // ========================================

    // Establishes WebSocket connection with automatic reconnection
    function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

        socket.addEventListener("open", (event) => {
            isConnected = true;
            console.log("Connected to WebSocket server");
        });

        socket.addEventListener("message", (event) => {
            const message = event.data;
            handleIncomingMessage(message);
        });

        socket.addEventListener("close", (event) => {
            isConnected = false;
            console.log("Disconnected from WebSocket server");

            // Auto-reconnect after 3 seconds
            setTimeout(connectWebSocket, 3000);
        });

        socket.addEventListener("error", (error) => {
            console.error("WebSocket error:", error);
        });
    }

    // Handles incoming WebSocket messages
    // @param {string} message - The received message
    function handleIncomingMessage(message) {
        // Remove existing typing indicator
        const typingIndicator = document.querySelector(
            ".typing-indicator-container"
        );
        if (typingIndicator) {
            chatMessages.removeChild(typingIndicator);
        }

        if (message === "Bot is thinking...") {
            appendTypingIndicator();
        } else {
            appendBotMessage(message);
            scrollToBottom();
        }
    }
    // ========================================
    // MESSAGE FUNCTIONS
    // ========================================

    /**
     * Sends a message through WebSocket
     */
    function sendMessage() {
        const message = userInput.value.trim();

        if (message && isConnected) {
            appendUserMessage(message);
            userInput.value = "";
            socket.send(message);
            scrollToBottom();
        }
    }

    /**
     * Appends a user message to the chat
     * @param {string} message - The message text
     */
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

    /**
     * Appends a bot message to the chat
     * @param {string} message - The message text
     */
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

    /**
     * Displays a typing indicator in the chat
     */
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

    // ========================================
    // UTILITY FUNCTIONS
    // ========================================

    /**
     * Formats message text with HTML links and line breaks
     * @param {string} message - The raw message text
     * @returns {string} - The formatted HTML string
     */
    function formatMessage(message) {
        let escapedMessage = escapeHTML(message);

        // Convert URLs to clickable links
        escapedMessage = escapedMessage.replace(
            /(https?:\/\/[^\s]+)/g,
            '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
        );

        // Convert line breaks to HTML breaks
        escapedMessage = escapedMessage.replace(/\n/g, "<br>");

        return escapedMessage;
    }

    /**
     * Escapes HTML characters to prevent XSS attacks
     * @param {string} text - The text to escape
     * @returns {string} - The escaped text
     */
    function escapeHTML(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Scrolls the chat to the bottom to show latest messages
     */
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ========================================
    // PDF UPLOAD FUNCTIONS
    // ========================================

    /**
     * Handles PDF file upload to the server
     * @param {File} file - The PDF file to upload
     */
    async function uploadPDF(file) {
        appendBotMessage(`Uploading PDF "${file.name}"...`);
        scrollToBottom();

        const formData = new FormData();
        formData.append("pdf", file);

        try {
            const response = await fetch("/upload-pdf", {
                method: "POST",
                body: formData,
            });

            const result = await response.json();

            if (response.ok) {
                appendBotMessage(
                    `PDF "${file.name}" uploaded successfully! The document contains ${result.pages} pages and is now available for questions.`
                );
            } else {
                appendBotMessage(
                    `Error uploading PDF: ${
                        result.detail || "Unable to make HTTP request"
                    }`
                );
            }
        } catch (error) {
            appendBotMessage(`Error uploading PDF: ${error.message}`);
        }

        // Clear the file input and scroll to bottom
        pdfUpload.value = "";
        scrollToBottom();
    }

    // ========================================
    // INITIALIZATION & EVENT LISTENERS
    // ========================================

    /**
     * Initialize WebSocket connection
     */
    connectWebSocket();

    /**
     * Set up event listeners for user interactions
     */
    // Send button click
    sendBtn.addEventListener("click", sendMessage);

    // PDF upload button click
    uploadBtn.addEventListener("click", () => {
        pdfUpload.click();
    });

    // File selection handler
    pdfUpload.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            uploadPDF(file);
        }
    });

    // Enter key to send message (Shift+Enter for new line)
    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    // Focus the input field when page loads
    userInput.focus();
});
