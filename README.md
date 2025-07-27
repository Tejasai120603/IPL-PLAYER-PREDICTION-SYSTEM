# 🏏 IPL Player Performance Predictor

This project is a web application built with Streamlit that predicts an IPL player's batting performance across the three main phases of a T20 match (Powerplay, Middle Overs, and Death Overs). The application uses an XGBoost regression model trained on historical IPL data to forecast the number of runs a player is likely to score in each phase.

## ✨ Key Features

-   **Dynamic Player Selection**: Enter any player's name and their team to get customized predictions.
-   **Phase-Wise Prediction**: The model provides separate run predictions for each of the three match phases.
-   **Data-Driven Insights**: Moves beyond simple career averages to provide context-aware predictions based on various match features.
-   **Performance Visualization**: Includes scatter plots to compare the model's predicted runs against the player's actual runs for each phase.
-   **Interactive UI**: A simple and clean user interface powered by Streamlit for easy interaction.
## ⚙️ Setup and Usage

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install the required libraries:**
    It's recommended to use a virtual environment.
    ```bash
    pip install streamlit pandas numpy matplotlib xgboost scikit-learn
    ```

3.  **Run the Streamlit app:**
    Navigate to the project directory in your terminal and run:
    ```bash
    python -m streamlit run app.py
    ```
    The application will open in your web browser.

## 📊 Sample Outputs

Here are some examples of the application's output.

### Main Interface

<img width="975" height="819" alt="Screenshot 2025-07-27 160332" src="https://github.com/user-attachments/assets/ff7c8861-ebba-49a9-9d85-8701f731a5e8" />


### Prediction Results Table

<img width="935" height="661" alt="Screenshot 2025-07-27 160403" src="https://github.com/user-attachments/assets/d42fe92f-4060-434e-8b41-f31c713f91bf" />


### Performance Visualization

<img width="892" height="533" alt="Screenshot 2025-07-27 160424" src="https://github.com/user-attachments/assets/78dfe322-f024-4eb5-b278-d1c60016c885" />

## 📞 Contact

For inquiries or collaboration, feel free to reach out at `bl.en.u4eac22079@bl.students.amrita.edu`.

