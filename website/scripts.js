const sendButton = document.getElementById('sendButton');
    const rangeInput = document.getElementById('rangeInput');
    const numberInput = document.getElementById('numberInput');
    const messageInput = document.getElementById('message');

    const endpoint = "http://localhost:5501/complete"; 

    rangeInput.addEventListener('input', () => {
        numberInput.value = rangeInput.value;
    });

    numberInput.addEventListener('input', () => {
        if (numberInput.value >= 5 && numberInput.value <= 100) {
            rangeInput.value = numberInput.value;
        }
    });

    // Function to send POST request with message and maxLength
    function sendRequest(message, maxLength) {
        loader.style.display = 'block';  // Show the loader when the request starts
        message = message.replace(/\s+$/, '');
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                maxLength: maxLength
            }),
        })
        .then(response => response.json())
        .then(data => {
            const serverResponse = data.response;
            appendToContenteditable(serverResponse);
        })
        .catch((error) => console.error('Error:', error))
        .finally(() => {
            loader.style.display = 'none';  // Hide the loader when the request completes (success or error)
        });
    }

    function appendToContenteditable(serverResponse) {
        const userMessage = document.getElementById('message').innerHTML; // Get current content of the div
        
        // Use backticks correctly for template literals
        const highlightedResponse = `<span class="highlighted-text">${serverResponse}</span>`;

        document.getElementById('message').innerHTML = userMessage + " " + highlightedResponse;
        clearButton.disabled = false;
    }

    sendButton.addEventListener('click', (event) => {
    event.preventDefault(); // Prevent page reload

    const message = messageInput.innerText; 
    const maxLength = rangeInput.value; 

    // Validate that message is not empty
    if (message.trim() === "") {
        alert("Please enter a message.");
        return;
    }

    sendRequest(message, maxLength);
});


const clearButton = document.getElementById('clearButton'); // Get the Clear button

clearButton.addEventListener('click', () => {
    // Find all the highlighted text spans and remove them
    const highlightedSpans = document.querySelectorAll('.highlighted-text');
    highlightedSpans.forEach(span => {
        span.remove(); 
    });

    // After clearing, disable the Clear button again
    clearButton.disabled = true;
});