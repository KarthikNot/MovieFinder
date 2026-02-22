# MovieFinder

MovieFinder is a movie recommendation system that uses content-based filtering to suggest movies based on user preferences. The project is built using Python and Streamlit, and it leverages machine learning techniques to provide personalized movie recommendations.

## Features
- Content-based movie recommendation system
- Preprocessing and vectorization of movie datasets
- Interactive web interface using Streamlit
- Dockerized application for easy deployment

## Project Structure
```
MovieFinder/
├── artifacts/                # Directory for storing artifacts like vectorized matrices
├── data/                     # Contains raw and preprocessed datasets
├── notebooks/                # Jupyter notebooks for data preprocessing and experimentation
├── src/                      # Source code for the application
│   ├── components/           # Modules for data preprocessing and vectorization
│   ├── constants/            # Constants used across the project
│   ├── logger/               # Logging utilities
│   └── pipelines/            # Training and recommendation pipelines
├── Dockerfile                # Docker configuration for the application
├── compose.yaml              # Docker Compose configuration
├── requirements.txt          # Python dependencies
├── setup.py                  # Setup script for the project
├── server.py                 # Streamlit server for the web interface
└── README.md                 # Project documentation
```

## Installation

### Prerequisites
- Python 3.10
- Docker

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MovieFinder
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application locally:
   ```bash
   python src/pipelines/training_pipeline.py
   streamlit run server.py
   ```

## Docker Setup
1. Build the Docker image:
   ```bash
   docker-compose build
   ```
2. Run the Docker container:
   ```bash
   docker-compose up
   ```

The application will be available at `http://localhost:8601`.

## Usage
1. Open the application in your browser.
2. Select a movie from the dropdown menu.
3. Get personalized movie recommendations.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your fork.
4. Submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [TMDB](https://www.themoviedb.org/) for the movie dataset.
- The open-source community for providing amazing tools and libraries.
