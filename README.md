# Centralized AI System

This repository contains the code for a centralized AI system designed for efficient task processing and modular integration. The architecture enables seamless interactions among various AI components.

## Project Structure

- `app/`: Contains the main application logic and AI processing modules.
- `tests/`: Includes unit tests for validating the functionality of the system.
- `requirements.txt`: Lists Python dependencies for the project.
- `Dockerfile`: Provides the setup to containerize the application.
- `download_models.py`: Script to download required AI models.
- `README.md`: This file, containing an overview of the project.

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:
- Python 3.8 or later
- Docker (if you plan to use containerization)
- A virtual environment manager (e.g., `venv`, `virtualenv`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rlombard/centralized-ai.git
   cd centralized_ai
   ```

2. Install the required Python packages:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Download the necessary models:
   ```bash
   python download_models.py
   ```

### Running the Application

Run the application locally:
```bash
python app/main.py
```

### Docker Setup

1. Ensure models are downloaded:
   ```bash
   python download_models.py
   ```

2. Build the Docker image:
   ```bash
   docker build -t centralized-ai .
   ```

3. Run the container:
   ```bash
   docker run -d -p 8000:8000 centralized-ai
   ```

## Features

- Modular AI task processing
- Integration with external data sources
- Model management and updates

## API Endpoints

The following endpoints are available:

1. **Health Check**
   ```bash
   curl -X GET http://localhost:8000/health
   ```

2. **Tag Image**
   ```bash
   curl -X POST http://localhost:8000/tag-image \
   -H "Content-Type: multipart/form-data" \
   -F "file=@path/to/your/image.jpg"
   ```

3. **Describe Image**
   ```bash
   curl -X POST http://localhost:8000/describe-image \
   -H "Content-Type: multipart/form-data" \
   -F "file=@path/to/your/image.jpg"
   ```

4. **Classify Image**
   ```bash
   curl -X POST http://localhost:8000/classify-image \
   -H "Content-Type: multipart/form-data" \
   -F "file=@path/to/your/image.jpg"
   ```

5. **Analyze Text**
   ```bash
   curl -X POST http://localhost:8000/analyze-text \
   -H "Content-Type: application/json" \
   -d '{"text": "Your text here"}'
   ```

### Testing via `curl`
Use these commands to verify that the endpoints are working correctly after starting the server.

## Contributing

Contributions are welcome! Please follow the steps below:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes and push to your forked repository.
4. Open a pull request to the main repository.

## License

This project is fully open source and free to use as is.

## Acknowledgments

- Special thanks to all contributors.
- Libraries and frameworks used in this project.

---
For questions or suggestions, please create a discussion on GitHub
