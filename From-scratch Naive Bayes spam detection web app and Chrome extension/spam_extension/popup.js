// Add click listener to the "Analyze" button
document.getElementById('check').addEventListener('click', async () => {

    // Get the user-entered message from the textarea
    const msg = document.getElementById('input').value;

    // Element where we will display the prediction result
    const resEl = document.getElementById('result');

    try {
        // Send POST request to Flask backend with the message to classify
        const resp = await fetch('http://127.0.0.1:2026/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        });

        // If server responds with a non-200 code, treat it as an error
        if (!resp.ok) throw new Error(`Server responded ${resp.status}`);

        // Convert server JSON response into a JavaScript object
        const data = await resp.json();

        // Determine whether the returned prediction is "Spam"
        const isSpam = data.prediction === 'Spam';

        // Update the UI with the model’s prediction
        resEl.textContent = data.prediction;

        // Apply a CSS class so the text is styled red (spam) or green (not spam)
        resEl.className = isSpam ? 'spam' : 'not-spam';

    } catch (e) {
        // Handle fetch errors
        console.error(e);

        // Display an error message in the UI
        resEl.textContent = 'Could not connect to Flask server.';

        // Remove any styling class
        resEl.className = '';
    }
});
