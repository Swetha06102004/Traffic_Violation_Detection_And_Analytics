                                                                                Traffic Violation Detection and Analytics
  An intelligent web-based tool built using Streamlit that analyzes traffic violation data to detect potential fraud patterns and provides visual summaries and risk classifications based on location.
Features:
        * Upload traffic violation datasets in csv format. 
        * Predict fradulant violations using a trained ML model.
        * Visualize fraud risk percentage by location(High/Medium/Low).
        * View tabular summaries of total records,fraud cases and risk levels.
        * Download risk summary as csv.
        * Beautiful UI with custom background and bar chart visualizations.
ML Model Used:
        Model: 
                * RandomForestClassifier(trained on violation patterns)
        Input Features:
                * Violation Type
                * Fine Amount
                * Speed Limit
                * Recorded Speed
                * Alcohol Level
                * Driver Age
                * Vehicle Type
                * Previous Violations
                * Fine paid
        Encoders:
                Label Encoders - Violation-Type,Vehicle_Type,Fine_Paid.
Project Structure:
                Traffic_Violation_Detection_And_Analytics/
                ├── coding/
                │   └── app.py
                │   └── bg.jpg
                ├── models/
                │   └── fraud_detection_model.pkl
                ├── encoders/
                │   └── le_violation_type.pkl
                │   └── le_vehicle_type.pkl
                │   └── le_fine_paid.pkl
                ├── datasets/
                │   └── sample_data.csv
Demo:
        Live Streamlit App: https://trafficviolationdetectionandanalytics-3xlbkrnhjxeab6e4mrds5y.streamlit.app/
Dataset Used:
       * Based on synthetic or anonymized traffic violation datasets.
       * Custom csv files can be uploaded(must contain relevant columns like Violation_Type,Location,etc.)            
Author:
       Swetha Chandrasekaran
       GitHub:  @Swetha06102004
       Project  submitted for academic + placement purpose
          
          
