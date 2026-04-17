# Spam Detector Capstone Project

This repository contains a comprehensive spam detection system consisting of a Flask web application and a Chrome browser extension. This project provides users with multiple ways to identify spam content using a trained Naive Bayes model.

## Project Overview

This repository contains two main components in separate folders:

- **Flask Web Application**: A standalone web interface for spam detection
- **Chrome Extension**: A browser extension that integrates spam detection directly into your browser

## Features

- Real-time spam detection
- User-friendly web interface
- Chrome extension for seamless integration
- Local deployment

## Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.7 or higher
- Visual Studio Code (recommended)
- Google Chrome browser
- pip (Python package installer)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install Python Dependencies

Navigate to the project root directory and install the required packages:

```bash
pip install -r requirements.txt
```

This will install Flask, flask-cors, numpy, and other necessary dependencies.

## Running the Flask Web Application

1. Open Visual Studio Code
2. Open the `spam_app` folder in VS Code
3. Open a new terminal in VS Code (Terminal → New Terminal)
4. Run the Flask application:
   ```bash
   python app.py
   ```
5. The terminal will display a localhost URL
6. Open your web browser and navigate to the provided localhost URL
7. The spam detection web interface should now be accessible

## Running the Chrome Extension

### Setting Up the Extension Backend

1. Open Visual Studio Code
2. Open the `spam_extension` folder in VS Code
3. Open a new terminal in VS Code
4. Run the Flask backend for the extension:
   ```bash
   python app.py
   ```
5. Keep this terminal running in the background

### Installing the Chrome Extension

1. Open Google Chrome
2. Navigate to `chrome://extensions` in your address bar
3. Enable "Developer mode" using the toggle in the top-right corner
4. Click "Load unpacked"
5. Browse to and select the `spam_extension` folder from this repository
6. The extension should now appear in your Chrome extensions bar
7. You can now use the extension while browsing

## Usage

### Web Application
- Navigate to the Flask app URL
- Enter text content you want to check
- Receive spam classification results

### Chrome Extension
- Click the extension icon in your Chrome toolbar
- Use the interface to analyze content on the current page or input custom text
- View spam detection results directly in your browser

## Troubleshooting

- **Port already in use**: If you see an error about the port being in use, modify the port number in `app.py`
- **Module not found errors**: Verify all required Python packages are installed using pip

## Acknowledgments

This project was completed as part of my M.S. Data Science capstone project to demonstrate integration of classical machine learning techniques with modern application deployment and user interaction design.

If you have questions, feel free to reach out!
