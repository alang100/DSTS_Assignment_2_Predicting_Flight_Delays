# Predicting Flight Delays

## DSTS Final Project

This project consists of the following files:
1. Part A - onpremises.ipynb
2. Part B - oncloud.ipynb # Note this is just the unrun code.
3. Part B - oncloud.html
4. A tableau dashboard was made for this project which is available on the following link: https://public.tableau.com/app/profile/alan.gaugler/viz/DSTS_2/Delays?publish=yes

### Part A

The dataset used for this project is a zip file of 60 zip files. Each of these 60 zip files contain flight data of US domestics flight for each month in the 5 year period from January 2014 to December 2018. Included are the following files:
1. A CSV file containing all the flight information in the US for a one-month period. This will contain several hundred thousand rows for each flight and 109 variables. Many of these variables are partially or mostly empty.
2. A ‘readme.html’ file which consists of the field names and a brief description of them.
The flight data for this project was made available at the following site. Permissions will be required to access it. The entire dataset is 1.5 GB in size, so it cannot be uploaded here.

Similar data can be collected at the Bureau of Transportation Statistics at the following link:
https://www.transtats.bts.gov/

The ‘onpremises.ipynb’ file can be downloaded from this repository and run on any personal computer with a notebook installed. This file was run and modified on Jupyterlab in the Anaconda environment in Windows 11 but can be run in any other notebook environment.
To run it, the following packages must be installed in your environment.
* Pathlib
* Zipfile
* Pandas
* Numpy
* Matplotlib
* Seaborn

Other libraries must also be imported. These are part of the Python installation and will be imported upon executing the code.

Running the code is straightforward.
Place the code in your desired working directory and confirm the working directory is set to where the code is placed. 
A subfolder called ‘zipped_flight_data’ needs to be created. 
All the zip files for each month (described above) need to be placed within this folder. 
Executing this program will create a new sub folder ‘flight_data’ in the working directory and place all the unzipped CSV files into this folder as well as one unzipped ‘readme.html’ file. 
Three files will be created from this program and placed into the working directory.
1. ‘combined_files.csv’ – This consists of the 20 most important variables for modelling for this dataset which is filtered for the 9 busiest origination and destination airports and the 5 most popular airlines.
2. 'combined_csv_v1.csv' – This is a modification of the dataset and the file ‘combined_files.csv’ in which all the categorical features are one-hot encoded, so it consists of many more columns. 
3. 'combined_csv_v2.csv' – This is similar to 'combined_csv_v1.csv', however the weather data for each date at the airport has been added. Collecting this weather data is described below.

Weather data must be downloaded from the following site:
https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00023174,USW00012960,USW00003017,USW00094846,USW00013874,USW00023234,USW00003927,USW00023183,USW00013881&dataTypes=AWND,PRCP,SNOW,SNWD,TAVG,TMIN,TMAX&startDate=2014-01-01&endDate=2018-12-31.
This consists of weather data from the weather stations nearest to the 9 busiest airports during the period of the flight data. In this case January 2014 to December 2018. If you are running this program for a different period of time, please collect the weather data for the appropriate period and site locations from the same website.
This describes the instructions for running this code on other machines. It should be very simple to execute.
 


---

## Summary Report of Part B: Using AWS Sagemaker to Predict Airplane Delays Using Machine Learning Techniques

### Objective
The goal of Part B of this project was to build machine learning models to predict whether a flight would be delayed due to weather conditions. The focus was on flights departing from or arriving at the busiest domestic airports in the U.S., leveraging historical flight and weather data to enhance customer experience by providing delay predictions during flight bookings.

### Dataset
The dataset, provided by the Bureau of Transportation Statistics (BTS), consisted of detailed flight performance data for domestic U.S. flights from 2014 to 2018. The dataset contained features such as flight schedules, distances, origins, destinations, airlines, and weather information. Two combined datasets were created:
- **Dataset v1**: Contained core flight details.
- **Dataset v2**: Expanded dataset including additional weather and holiday information.

Both datasets had an imbalance in the target variable:
- 79% of flights were "No Delay" (class 0).
- 21% were "Delay" (class 1).

### Methodology
The project was implemented in Amazon SageMaker using the following steps:

#### Data Preparation
- Loaded datasets and verified integrity.
- Split the data into training (70%), validation (15%), and testing (15%) sets.
- Stratified splits ensured balanced representation of the target variable across subsets.

#### Model Building
- **Linear Learner Model**: Built a binary classification model using AWS's linear learner. The mini-batch size was increased to 1000 to improve training speed.
- **XGBoost Model**: Developed an ensemble model optimized for binary classification with hyperparameters tuned for boosting rounds and evaluation using AUC.

#### Evaluation Metrics
- Accuracy, precision, recall, F1-score, specificity, and AUC (Area Under the Curve) metrics were calculated for model evaluation.
- Confusion matrices and ROC curves were plotted to analyze performance.

#### Threshold Adjustment
- The default classification threshold (0.5) was reduced to 0.3 for the XGBoost model to improve recall for the minority class (delays).

### Results

#### Linear Learner Model (v1 Dataset)
- **Accuracy**: 79.01%
- **Recall for delays**: 0.13%
- **F1-score**: 0.25%
- **AUC**: 0.50
- The model was biased towards the majority class ("No Delay"), failing to accurately predict delays.

#### XGBoost Model
- **Dataset v1**:
  - **Accuracy**: 79.16%
  - **Recall for delays**: 1.67%
  - **F1-score**: 3.25%
  - **AUC**: 0.51
- **Dataset v2**:
  - **Accuracy**: 80.27%
  - **Recall for delays**: 11.97%
  - **F1-score**: 20.29%
  - **AUC**: 0.55
- The inclusion of weather and holiday data significantly improved recall and overall predictive power.

#### Threshold Adjustment for XGBoost (v2 Dataset)
- Lowering the threshold to 0.3 improved recall for delays from 11.97% to **40.53%**, albeit at the cost of overall accuracy (**77.41%**) and precision (**45.70%**). 
- The F1-score improved to **42.96%**, reflecting a better balance between precision and recall.

### Conclusions

#### Model Comparison
- XGBoost outperformed the linear learner model in all key metrics, particularly for predicting the minority class ("Delay").
- Dataset v2 (with additional weather and holiday features) enhanced the model's predictive ability.

#### Key Trade-offs
- Adjusting the classification threshold from 0.5 to 0.3 drastically improved the recall for delays but reduced precision and overall accuracy. This trade-off highlights the importance of business context when optimizing classification thresholds.

#### Recommendations
- Further hyperparameter tuning, including grid search, could improve model performance, though it requires extended processing time.
- Additional features, such as airport-specific congestion data or real-time weather updates, could further enhance prediction accuracy.
- Future work could explore alternative ensemble models or deep learning approaches for better handling class imbalances.

### Impact
This project demonstrated the potential of machine learning in identifying flight delays due to weather conditions. While further improvements are necessary for real-world deployment, the insights gained establish a foundation for developing robust predictive tools for the airline industry. 

This project also provided me with excellent experience in utilizing AWS Sagemaker.
