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
    const pdfUpload = document.getElementById("pdf-upload");
    const pdfPlaceholder = document.getElementById("pdf-placeholder");
    const divider = document.getElementById("divider");
    const pdfSection = document.getElementById("pdf-section");
    const mainContainer = document.querySelector(".main-container");
    const pdfCanvas = document.getElementById("pdf-canvas");
    const prevPageBtn = document.getElementById("prev-page");
    const nextPageBtn = document.getElementById("next-page");
    const pageInfo = document.getElementById("page-info");
    const pageInput = document.getElementById("page-input");
    const zoomOutBtn = document.getElementById("zoom-out");
    const zoomInBtn = document.getElementById("zoom-in");
    const zoomLevel = document.getElementById("zoom-level");

    // ========================================
    // RESIZABLE DIVIDER FUNCTIONALITY
    // ========================================
    let isResizing = false;

    divider.addEventListener("mousedown", (e) => {
        isResizing = true;
        document.body.style.cursor = "col-resize";
        document.body.style.userSelect = "none";
        e.preventDefault();
    });

    document.addEventListener("mousemove", (e) => {
        if (!isResizing) return;

        const containerRect = mainContainer.getBoundingClientRect();
        const newPdfWidth =
            ((e.clientX - containerRect.left) / containerRect.width) * 100;

        // Constrain width between 20% and 80%
        if (newPdfWidth >= 20 && newPdfWidth <= 80) {
            pdfSection.style.flex = `0 0 ${newPdfWidth}%`;
        }
    });

    document.addEventListener("mouseup", () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = "";
            document.body.style.userSelect = "";
        }
    });

    // ========================================
    // GLOBAL STATE
    // ========================================
    let socket = null;
    let isConnected = false;

    // PDF display state
    let currentPdf = null;
    let currentPage = 1;
    let totalPages = 0;
    let currentZoom = 1.0;

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
    // PDF DISPLAY FUNCTIONS
    // ========================================

    /**
     * Loads and displays a PDF from the given URL
     * @param {string} pdfUrl - URL to the PDF file
     */
    async function loadPDF(pdfUrl) {
        try {
            // Configure PDF.js worker
            pdfjsLib.GlobalWorkerOptions.workerSrc =
                "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

            // Load the PDF
            const pdf = await pdfjsLib.getDocument(pdfUrl).promise;
            currentPdf = pdf;
            totalPages = pdf.numPages;
            currentPage = 1;

            // Update page info
            updatePageInfo();

            // Show the PDF canvas and hide placeholder
            pdfCanvas.style.display = "block";
            pdfPlaceholder.style.display = "none";

            // Render the first page
            await renderPage(currentPage);
        } catch (error) {
            console.error("Error loading PDF:", error);
            appendBotMessage(`Error loading PDF: ${error.message}`);
        }
    }

    /**
     * Renders a specific page of the PDF
     * @param {number} pageNum - Page number to render
     */
    async function renderPage(pageNum) {
        if (!currentPdf || pageNum < 1 || pageNum > totalPages) return;

        try {
            const page = await currentPdf.getPage(pageNum);
            const viewport = page.getViewport({ scale: currentZoom });

            // Set canvas dimensions
            const context = pdfCanvas.getContext("2d");
            pdfCanvas.height = viewport.height;
            pdfCanvas.width = viewport.width;

            // Render the page
            const renderContext = {
                canvasContext: context,
                viewport: viewport,
            };

            await page.render(renderContext).promise;
            currentPage = pageNum;
            updatePageInfo();
        } catch (error) {
            console.error("Error rendering page:", error);
        }
    }

    /**
     * Updates the page information display
     */
    function updatePageInfo() {
        pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
        zoomLevel.textContent = `${Math.round(currentZoom * 100)}%`;

        // Update page input
        pageInput.value = currentPage;
        pageInput.max = totalPages;

        // Update button states
        prevPageBtn.disabled = currentPage <= 1;
        nextPageBtn.disabled = currentPage >= totalPages;
    }

    /**
     * Goes to the previous page
     */
    function goToPreviousPage() {
        if (currentPage > 1) {
            renderPage(currentPage - 1);
        }
    }

    /**
     * Goes to the next page
     */
    function goToNextPage() {
        if (currentPage < totalPages) {
            renderPage(currentPage + 1);
        }
    }

    /**
     * Zooms in the PDF
     */
    function zoomIn() {
        currentZoom = Math.min(currentZoom + 0.25, 3.0);
        renderPage(currentPage);
    }

    /**
     * Zooms out the PDF
     */
    function zoomOut() {
        currentZoom = Math.max(currentZoom - 0.25, 0.5);
        renderPage(currentPage);
    }

    /**
     * Goes to a specific page number
     * @param {number} pageNum - The page number to navigate to
     */
    function goToPage(pageNum) {
        const page = parseInt(pageNum);
        if (page >= 1 && page <= totalPages && page !== currentPage) {
            renderPage(page);
        }
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
                    `PDF "${file.name}" uploaded successfully! The document contains ${result.pages} pages.`
                );

                // Load and display the PDF
                if (result.pdf_url) {
                    await loadPDF(result.pdf_url);
                }
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

    // PDF placeholder click (replaces upload button)
    pdfPlaceholder.addEventListener("click", () => {
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

    // PDF control event listeners
    prevPageBtn.addEventListener("click", goToPreviousPage);
    nextPageBtn.addEventListener("click", goToNextPage);
    zoomInBtn.addEventListener("click", zoomIn);
    zoomOutBtn.addEventListener("click", zoomOut);

    // Page input event listeners
    pageInput.addEventListener("change", (event) => {
        goToPage(event.target.value);
    });

    pageInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            goToPage(event.target.value);
            event.target.blur(); // Remove focus from input
        }
    });

    // Focus the input field when page loads
    userInput.focus();
});
