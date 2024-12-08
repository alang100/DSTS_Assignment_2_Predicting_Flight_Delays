# Predicting Flight Delays

## DSTS Final Project

This project focused on forecasting flight delays at major U.S. airports and was divided into two distinct phases:

- **Part A**: Conducted using Python in Jupyter Notebooks, this phase involved data preprocessing, exploratory data analysis (EDA), and initial model development on a local machine. The emphasis was on understanding the dataset, creating features, and training baseline machine learning models.

- **Part B**: Carried out on the AWS Cloud platform and its SageMaker service, a fully managed machine learning service that enables scalable model training and deployment. This phase focused on scaling the data and model workflows, training advanced models like XGBoost, and evaluating performance metrics. SageMaker's integration with cloud resources facilitated the efficient handling of large datasets and model optimization.

By combining on-premises development with cloud-based solutions, the project showcased a comprehensive approach to solving real-world machine learning problems. The use of SageMaker enabled scalable model training, efficient resource utilization, and streamlined evaluation in a production-ready environment.

This project consists of the following files:
1. [Part A - onpremises.ipynb](Part%20A%20-%20Predicting%20Flight%20Delays.ipynb) code with output.
2. [Part A - onpremises.html](Part%20A%20-%20Predicting%20Flight%20Delays.html) code with output as a HTML file.
3. [Part B - oncloud.ipynb](Part%20B%20-%20AWS%20oncloud%20code.ipynb)  Note. This is the code that was written, but it does not contain the output. 
4. [Part B - oncloud.html](Part%20B%20-%20oncloud.html) output and report of the code run on AWS Sagemaker.
5. [Flights Stats Tableau Dashboard](https://public.tableau.com/app/profile/alan.gaugler/viz/DSTS_2/Delays?publish=yes) highligthing several visualations of the flight data

## Part A

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
 
# Summary Report: Predicting Flight Delays (Part A)

## Objective
The objective of Part A was to develop machine learning models to predict flight delays at major U.S. airports based on historical data. Looking at the information of all flights available for each month, it is clear to note that there is a lot of information available with 1.5 GB of flight information across several years (2014 to 2018). There are several variables about each flight for many airlines across the entire nation (USA).
Flight delays are influenced by many variables including, gate and runway congestion, air traffic, maintenance and weather. These will vary from airport to airport. The busiest airports such as Chicago O'Hare and Atlanta may have more flight delays due to congestion and airports in the north of the country may have more flight delays of cancellations due to weather conditions such as snow. The objective of this problem is to predict if a flight will be delayed by 15 minutes or more. Machine learning is well suited to predicting this for the reasons described above. A lot of data across the entire nation and a lot of detailed information about the flights (features or variables) ensure that machine learning can be applied to this problem.

**Business Problem**  
Stakeholders in the airline industry can identify where and when the most delays take place by applying machine learning to predict the delays based on the historical data. Knowing where and when the delays take places, the airports and airlines can investigate further what causes these delays and can focus in these areas to improve their processes and methods to improve flight delays in these areas. This will be a better outcome for their customers (passengers and freight companies) which will ultimately help in retaining their customer base or increasing it with better service. This will of course improve their profitability.

## Methodology
### Data Preparation
- **Source Data:**
  - Historical flight performance data from the Bureau of Transportation Statistics, covering domestic flights in the U.S. from 2014 to 2018.
- **Feature Engineering:**
  - **Time Features:** Extracted variables such as the day of the week, month, and quarter to analyze seasonal trends.
  - **Weather Data:** Enriched flight data with metrics such as snow, precipitation, average wind speed, and temperature for origin and destination airports.
  - **Holiday Data:** Added features indicating proximity to major U.S. holidays.
  - **Flight Metrics:** Included variables such as flight distance, departure hour, and arrival hour.
- **Data Cleaning:**
  - Removed duplicate and inconsistent records.
  - Handled missing values by imputation where necessary or removing rows/columns with excessive gaps.
- **Class Balancing:**
  - Highlighted a significant imbalance in the target variable, with "No Delay" accounting for 79% of the data and "Delay" for 21%.
  - Oversampling and undersampling techniques were considered but not fully applied in this project due to the limited timeframe.

### Exploratory Data Analysis (EDA)
- **Trends and Correlations:**
  - Identified patterns such as increased delays during peak travel times (e.g., holidays and weekends).
  - Observed correlations between weather variables (e.g., high precipitation or wind speed) and flight delays.
- **Visualization:**
  - Created several bar charts and other visualizations to understand the distribution of delays across airports, times, and weather conditions.
  - Highlighted key differences in delay rates between large hub airports and smaller regional ones.

### Model Development
1. **Baseline Model:**
   - Logistic regression was used to establish a baseline for predicting delays.
   - Key metrics such as accuracy, recall, and precision revealed the limitations of a linear model in handling the complexity of the dataset.
2. **Advanced Models:**
   - Another Logistic Regression model and a Random Forest model were developed utilizing the weathwer and holiday data to improve performance over the baseline model.
3. **Evaluation Strategy:**
   - The dataset was split into training (80%), and testing (20%) subsets.
   - Stratified splits ensured the class imbalance was preserved across subsets.
   - Metrics such as accuracy, precision, recall, F1-score, and AUC-ROC were used to evaluate performance, with a particular focus on recall for the "Delay" class to prioritize identifying delayed flights.

### Results

#### Logistic Regression Model 1:
- **Performance Overview:**
  - **Accuracy:** 79.02%, primarily reflecting the "No Delay" majority class.
  - **Recall for Delays:** Extremely low at 0.33%, indicating poor identification of delayed flights.
  - **Precision:** 54.07%, showing moderate reliability when predicting delays.
  - **F1-Score:** 0.65%, reflecting poor balance between precision and recall.
  - **AUC:** 0.50, equivalent to random guessing, confirming poor performance.


#### Logistic Regression Model 2 (with additional features):
- **Performance Overview:**
  - **Accuracy:** 78.99%, similar to Model 1.
  - **Recall for Delays:** Improved significantly to 4.18%.
  - **Precision:** Slightly lower at 49.57%.
  - **F1-Score:** Increased to 7.70%, reflecting better balance between precision and recall.
  - **AUC:** Slightly improved to 0.52, showing marginal gains in class separation.

#### Random Forest Model:
- **Performance Overview:**
  - **Accuracy:** 80.99%, the highest among the three models.
  - **Recall for Delays:** Increased substantially to 26.34%, showing the model's improved ability to identify delays.
  - **Precision:** Reached 60.89%.
  - **F1-Score:** Improved to 36.78%, indicating a better balance between precision and recall.
  - **AUC:** 0.61, reflecting enhanced separation of delayed and non-delayed flights.
  - Despite overfitting on the training data, the Random Forest model significantly outperformed the logistic regression models on the test set.


### Conclusion

1. **Model Performance:**
   - All three models struggled to predict delays effectively due to severe class imbalance, with predictions heavily skewed toward the "No Delay" majority class. 
   - The **Random Forest model** delivered the best performance, significantly improving recall and F1-score while maintaining the highest accuracy. However, the improvement came with signs of overfitting on the training data.

2. **Feature Engineering Impact:**
   - Adding features like weather conditions, holidays, and departure hour data led to noticeable improvements in identifying delays. Holidays and extreme weather conditions (e.g., heavy snow or rain) were particularly influential, as they are strongly correlated with delays.
   - One-hot encoding of categorical features, including departure hour and airport variables, also contributed to the improvement in model performance.

3. **Challenges:**
   - The **severe class imbalance** made it difficult to predict the minority class ("Delay"), resulting in poor recall for delays in the first two models.
   - Overfitting in the Random Forest model indicated the need for better regularization and model tuning.
   - Limited time for exploring advanced modeling techniques and hyperparameter optimization further constrained the potential performance.

4. **Future Steps:**
   - **Class Imbalance Techniques:** Implement SMOTE or similar oversampling methods to balance the dataset, enabling the model to prioritize both classes equally.
   - **Feature Enhancements:** Add more granular features, such as real-time weather updates, airport congestion levels, and operational disruptions (e.g., maintenance or construction activities).
   - **Advanced Models:** Experiment with models like Gradient Boosting Machines, Support Vector Machines, and Neural Networks to better capture non-linear relationships in the data.
   - **Hyperparameter Tuning:** Perform grid search or randomized search to optimize the models' configurations.
   - **Regularization:** Apply techniques to reduce overfitting in the Random Forest model, such as limiting tree depth or increasing the minimum samples per leaf.

5. **Key Learnings:**
   - Addressing class imbalance is critical, especially when predicting the minority class is vital for business outcomes.
   - Good data preparation, including feature engineering and scaling, is often more impactful than choosing the most advanced model.
   - Machine learning is an iterative process, requiring multiple rounds of experimentation, fine-tuning, and feature enhancement to achieve desired results.

In conclusion, while the Random Forest model showed promise with a significantly improved recall, further iterations with additional feature engineering, balancing techniques, and model tuning are essential to meet the business goal of accurately predicting flight delays.

---

## Part B
### Summary Report of Part B: Using AWS Sagemaker to Predict Airplane Delays Using Machine Learning Techniques

Provided here is a summary of the project for easy reference. Refer to the generated [Part B - oncloud.html](Part%20B%20-%20oncloud.html) report for more details.


### Objective
The goal of Part B of this project was to build machine learning models to predict whether a flight would be delayed due to weather conditions. The focus was on flights departing from or arriving at the busiest domestic airports in the U.S., leveraging historical flight and weather data to enhance customer experience by providing delay predictions during flight bookings.

### Dataset
The dataset, provided by the Bureau of Transportation Statistics (BTS), consisted of detailed flight performance data for domestic U.S. flights from 2014 to 2018. The dataset contained features such as flight schedules, distances, origins, destinations, airlines, and weather information. Two combined datasets were created:
- **Dataset v1**: Contained core flight details including schedules, distances, and airline information.
- **Dataset v2**: Expanded dataset including additional weather and holiday information.

Both datasets had an imbalance in the target variable:
- 79% of flights were "No Delay" (class 0).
- 21% were "Delay" (class 1).

### Methodology

#### Data Preparation
1. **Dataset Overview:**  
   Two datasets were utilized:  
   - **Dataset v1** contained core flight information, including flight schedules, distances, origins, destinations, and airlines.  
   - **Dataset v2** extended Dataset v1 by incorporating additional features such as weather conditions (e.g., precipitation, wind speed, snowfall) and holiday indicators, which were hypothesized to improve the model’s predictive performance.

2. **Data Loading and Integrity Verification:**  
   The datasets were loaded and verified for integrity. Missing values were handled appropriately, and feature engineering was conducted to ensure compatibility with the machine learning models. Key transformations included ensuring that the target variable ("Delay" or "No Delay") was in the first column as required by certain algorithms.

3. **Train-Test Split:**  
   - Each dataset was split into training (70%), validation (15%), and testing (15%) sets.  
   - Stratified sampling was used to preserve the distribution of the target variable across subsets, ensuring balanced representation of delayed and non-delayed flights.


#### Model Building
1. **Linear Learner Model:**  
   - AWS's linear learner algorithm was used as a baseline binary classification model.  
   - The mini-batch size was increased from 200 to 1000 to improve training efficiency without compromising accuracy.  
   - The model’s hyperparameters were set to optimize binary classification tasks, and training outputs were stored in Amazon S3 for subsequent evaluation.

2. **XGBoost Model:**  
   - XGBoost, an ensemble learning method, was implemented for its ability to handle complex relationships in the data and its effectiveness in addressing class imbalances.  
   - Key hyperparameters included the number of boosting rounds (42) and the evaluation metric (AUC).  
   - The model was trained on the same data pipeline as the linear learner for consistency and comparability.


#### Evaluation Metrics
- Both models were evaluated using standard metrics, including:  
  - **Accuracy:** Overall proportion of correct predictions.  
  - **Precision:** Proportion of correctly identified delays out of all predicted delays.  
  - **Recall (Sensitivity):** Proportion of actual delays correctly identified by the model.  
  - **F1-score:** The harmonic mean of precision and recall, providing a balanced measure of the model’s performance.  
  - **Specificity:** Proportion of correctly identified non-delays out of all non-delays.  
  - **AUC (Area Under the Curve):** Assesses the model’s ability to distinguish between the two classes.  
- Confusion matrices and ROC curves were generated for visual analysis of the model’s predictions.

#### Threshold Adjustment
- The classification threshold, initially set at 0.5, was adjusted to 0.3 for the XGBoost model using Dataset v2.  
- This adjustment aimed to improve recall for the minority class (delays) by predicting a higher proportion of flights as delayed. The trade-offs in accuracy, precision, and recall were carefully analyzed.


#### Comparative Analysis
- The performance of the models was compared across datasets (v1 and v2) and algorithms (linear learner vs. XGBoost).  
- The impact of additional features in Dataset v2 and the effect of threshold adjustment on the XGBoost model’s predictive capability were thoroughly evaluated.  
- The conclusions and recommendations were based on both quantitative metrics and the trade-offs observed during the evaluation process.


### Results

#### Linear Learner Model (v1 Dataset)
The linear learner model demonstrated significant limitations, achieving an accuracy of **79.01%**, which largely reflected the proportion of "No Delay" (majority class) instances in the dataset. The model struggled to identify delayed flights, with a recall of just **0.13%** for the delay class and an F1-score of **0.25%**, indicating poor balance between precision and recall. The AUC (Area Under the Curve) was **0.50**, suggesting that the model was no better than random guessing for this task.


#### XGBoost Model

**Dataset v1:**  
The XGBoost model outperformed the linear learner on the same dataset, achieving a slightly improved accuracy of **79.16%** and an F1-score of **3.25%**. It managed to predict more delayed flights, as indicated by an improved recall of **1.67%** and an AUC of **0.51**. However, these metrics still revealed a model biased toward the majority class, with significant room for improvement.

**Dataset v2:**  
The inclusion of weather and holiday data in Dataset v2 substantially improved the XGBoost model's performance. The accuracy increased to **80.27%**, and the model's ability to identify delays rose sharply, reflected in a recall of **11.97%** and an F1-score of **20.29%**. The AUC increased to **0.55**, highlighting the contribution of additional features to predictive accuracy.


#### Threshold Adjustment for XGBoost (v2 Dataset)
Lowering the classification threshold from **0.5** to **0.3** resulted in a trade-off:

- **Recall for delays** improved from **11.97%** to **40.53%**, significantly increasing the model's ability to identify delayed flights.  
- This improvement came at the cost of overall accuracy (**77.41%**), precision (**45.70%**), and specificity (**87.21%**), as more flights were incorrectly classified as delayed.  
- The F1-score improved to **42.96%**, showing a better balance between precision and recall, and the AUC rose to **0.64**, reflecting the model's enhanced ability to distinguish between delayed and non-delayed flights.

These results illustrate the trade-offs involved in optimizing a model for business-specific needs, such as prioritizing recall over precision when identifying flight delays.


### Brief Conclusions

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

### Detailed Conclsuion
This is the more detailed conclusion that I wrote in the report.

There are some notable differences between the linear and the ensemble methods. The linear model took fmore time to process than the XGBoost model, even though the pipeline is the same set up as the XGBoost model, apart from defining the model. I increased the mini batch size from 200 to 1000 and this improved the processing speed considerably withoout any loss in accuracy.

The linear model had similar performance to the on-premises logistic regression model. In the confusion matrix, very few ‘Delay’ classes were predicted (neither correctly nor incorrectly). The overall accuracy was very close to 79% or the percentage of ‘No Delay’ values in the target variable, the recall was very low and the AUC in the ROC curve was very close to 0.5, which is not a good result.

The ensemble XGBoost model performed considerably better than the linear learner on both datasets. The logical setting for the binary convert threshold is 0.5, which is what I set it too as default, meaning that if the output probability is less than 0.5 the predicted value is set to 0 or ‘no delay’. If it is greater than 0.5 then it is predicted as a 1 or a ‘delay’.

At a setting of 0.5, the model using combined_csv_v2.csv dataset performed better than for dataset v1. As was explained in the on-premises notebook, the extra features incorporated in v2, i.e., the holidays and the weather data, in particular heavy snow, rain or winds combined well to improve the accuracy of the second model. This was reflected in the metrics of the two models.

Model XGB_1 only predicted 859 Delays correctly, whereas XGB_2 predicted 6162, a significant improvement which is reflected in the improved recall, up from 1.67% to 11.97%. The overall accuracy and precision also improved slightly in the v2 dataset, as a result the specificity decline slightly from 99.74 to 98.41, but the overall measure, the F1-score increased significantly from 3.25% to 20.29%, a significant improvement. The AUC also had a good increase from 0.51 to 0.55. In spite of the improved metrics, I would consider this model to still be quite poor. In the classification report, The recall of the majority class is good at 98% but it is still poor for the monority class 'Delay' at 12%.

Many improvements could be tried, given more time. I have mentioned those in the conclusion to the on-premises notebook, so I won’t repeat them here in detail. One simple option to improve the performance of both the linear learner and the XGBoost model is hyperparameter tuning in a grid-search. This would however increase processing time significantly. A session is limited to only 2 hours which is not enough time for a proper grid-search.

As mentioned, another option in the on-premises solution is the binary convert threshold. As mentioned above, I set this to 0.5 which is the logical option. Looking at the last section of code, I change this to a setting of 0.3. This effectively predicts anything with a probability of greater than 0.3 as a 1 or a ‘delay’ class. This is manipulating the output data to increase the number of predictions of the minority class. Looking at the confusion matrices for a change in threshold from 0.5 to 0.3, the number of correctly predicted delays has increased starkly from 6162 to 20871, although this has come at a cost of fewer ‘no delays’ being correctly predicted, (190764 down to 169036). The overall accuracy has fallen from 80.27% to 77.41%, as has the precision (67.72% to 45.70%) and the specificity (98.41% to 87.21%), however the recall has risen sharply (11.97% to 40.53%) and the F1-score too (20.29% to 42.96%). The AUC in the ROC has also improved significantly from 0.55 to 0.64.

Manipulating this threshold is a trade-off of Recall vs precision and overall accuracy. It depends on how critical predicting the minority class is. In this case flight delays of 15 minutes or more are not very critical in my opinion, compared to say detecting cancer in a patient. More investigation would have to be done in setting this threshold, determining its importance in the business model and deciding on the optimal setting.

To conclude, the XGBoost is better than the linear learner (on-cloud) and the logistic regression models used in the on-premises notebook. It is better in all metrics over the other two models. The enhanced dataset of v2 with the added weather and holiday information also significantly improves the model’s accuracy in predicted flight delays.

 
