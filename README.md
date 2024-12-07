# DSTS_Assignment_2_Predicting_Flight_Delays

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
 
### Part B

To run the code for Part B on AWS is straight forward.
First you must have an AWS account.
Log into AWS. 

**Accesse the AWS Management Console**  
1.	At the top of these instructions, choose Start Lab to launch your lab. A Start Lab panel opens, which displays the lab status.
2.	Wait until you see the message Lab status: ready, then close the Start Lab panel by choosing the X.
3.	At the top of these instructions, choose AWS. This will open the AWS Management Console in a new browser tab. The system will automatically log you in.
4.	Arrange the AWS Management Console browser tab so that it displays next to these instructions. Ideally, you should be able to see both browser tabs at the same time, which can make it easier to follow the lab steps.
5.	To open JupyterLab: On the AWS Management Console, on the Services menu, choose Amazon SageMaker.
6.	From the navigation menu on the left, expand the Notebook section and choose Notebook instances.
7.	Click the orange ‘Create notebook instance’ button. Name the notebook. Select the instance type of ‘ml.m4.xlarge’ or better if you have sufficient resources.
8.	Open the additional configuration arrow and change Volume Size in GB to 25 (GB). Scroll down and click the orange ‘Create notebook instance button’.
9.	Under notebook instances you will see its status is ‘pending’. Once it changes to ‘InService’, on the right click the link ‘Open JupyterLab’.
10.	Wait for Jupyter lab to open. When it is opened click the upload button and select the ‘Part B - oncloud.ipynb’ from your hard drive as well as the two csv files that were created after executing part A. They are 'combined_csv_v1.csv' and 'combined_csv_v1.csv'. It will be faster to zip them up and unzip them in the Jupyterlab environment.
11.	Ensure the two CSV files are unzipped and placed in the default directory ‘SageMaker’
12.	Open the notebook and execute the code
