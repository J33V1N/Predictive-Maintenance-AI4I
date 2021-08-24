# Predictive-Maintenance-AI4I
Building ML model which can detect machine failure with given information. In this case, CNC machine

As Industry 4.0 continues to generate media attention, many companies are struggling with the realities of AI implementation. Indeed, the benefits of predictive maintenance such as helping determine the condition of equipment and predicting when maintenance should be performed, are extremely strategic. Needless to say that the implementation of ML-based solutions can lead to major cost savings, higher predictability, and the increased availability of the systems.

# Why Use ML Techinques for Predictive Maintenance
Frankly speaking, predictive maintenance doesn’t require anything more than an informal mathematical computation on when machine conditions are at a state of needed repair or even replacement so that maintenance can be performed exactly when and how is most effective.

**However, ML eliminates most of the guesswork and helps facility managers focus on other tasks, such as:**
- Create predictive models for maximizing assest lifetime, operational efficiency, or uptime. 
- Leverage past and continuous data.
- Optimize the periodic maintenance operations
- Aviod or minimize the downtimes. 

**Objective of this model** is to detect the machine failure using some of available information regarding the machine and also to build an interface.
## Tech Stack used:
- Python 3.7.4
- gradio 2.2.13
- pandas 1.2.3
- scikit-learn 0.23.2
- seaborn 0.11.1
- yellowbrick 1.3.post1
- matplotlib  3.4.2

# Dataset Information
Since real predictive maintenance datasets are generally difficult to obtain and in particular difficult to publish, we present and provide a synthetic dataset that reflects real predictive maintenance encountered in industry to the best of our knowledge.. 

From UCI machine learning repository:https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv

**Attribute Information:**
- UID: unique identifier ranging from 1 to 10000
- product ID: consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specific serial number
- air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
- process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
- rotational speed [rpm]: calculated from a power of 2860 W, overlaid with a normally distributed noise
- torque [Nm]: torque values are normally distributed around 40 Nm with a Ïƒ = 10 Nm and no negative values.
- tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a
- 'machine failure' label that indicates, whether the machine has failed in this particular datapoint for any of the following failure modes are true. 

# Machine Failure modes
**The machine failure consists of five independent failure modes:**
- tool wear failure (TWF): the tool will be replaced of fail at a randomly selected tool wear time between 200 to 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).
- heat dissipation failure (HDF): heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the tool's rotational speed is below 1380 rpm. This is the case for 115 data points.
- power failure (PWF): the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in our dataset.
- overstrain failure (OSF): if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints.
- random failures (RNF): each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in our dataset.

If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1. It is therefore not transparent to the machine learning method, which of the failure modes has caused the process to fail.

# Exploratory Data Analysis
As mentioned before there are five independent failure modes within system and also if there's failure it is flagged with "1" as failure is true and "0" as failure is false. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/83111155/130551895-e1ac3d16-55bd-430c-9a0b-bcb0147edf05.png" title="% failures">
  <p align="center">Percentage of machine failures</p>
</p>

In any well maintained systems, the occurances of failures are to be minimum in this case also that's true over only about **3.4%** of time the said machine fails. Let us see the share of different mode of faults. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/83111155/130551522-47a57741-97fb-4b28-9dda-0a2ec669695d.png" title="% failures bar chart">
  <p align="center">Percentage of different machine failures</p>
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/83111155/130554008-209e3a15-5a76-47b8-b76a-2e1daa5c1a1f.png" title="% failures pie chart">
  <p align="center">Pie chart of different failures</p>
</p>
Here Heat Dissipation failure seems to be more common type of failure within the system, thus this indicates that temperature monitoring of process and environment is to more important in order to minimize this type of failure also, it has 30.8% of share among other type of failures.
<p align="center">
  <img src="https://user-images.githubusercontent.com/83111155/130555717-6f710d0d-3d04-4fde-b3a0-7a8984e5da13.png" title="Corelation Plot">
  <p align="center">Correlation Plot of all attributes</p>
</p>
As expected air temperature has impact on process temperature thus, controlling air temperature will regulate the process temperature and reduce the possibilty of HDF failure.  

# Approach towards building classifiers
We are going to build classifiers with high precision rate than recall rate, because I want to minimize the **"false alarms"**. 

## Building Logisitic Regression Model 
Logisitic Regression model was build by using Air temp, process temp,rotational speed, torque and tool wear to predict the status of machine failure. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/83111155/130585703-f0f62926-d983-4a15-87c7-269ae875a785.png" title="Confusion Matrix of LR">
  <img src="https://user-images.githubusercontent.com/83111155/130585771-e1284ab7-414b-4017-86bf-bd24b70e8131.png" title="ROC Curve of LR">
  <p align="center">Confusion Matrix and ROC curve of logistic Regression</p>
</p>

From confusion matrix shows that 45 samples of class 1 were wrongly as Class 0 and only 16 samples were classified as Class 1. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/83111155/130585736-e61bcc33-77bb-465f-aee4-9ee3e1ebd042.png" title="PR Curve of LR">
  <p align="center">PR curve of logistic Regression</p>
</p>

With precision score at **0.62** and recall score at **0.26** logistic regression performed better well.But still to many **False** alarms. 

## Building Random Forest Classifier Model 
Same features are used in building the classifier and see the performance. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/83111155/130585707-71b37164-c91d-4359-9339-9da2d0cabef5.png" title="Confusion Matrix of RFC">
  <img src="https://user-images.githubusercontent.com/83111155/130585763-edff3129-d397-4016-902f-908f8896efba.png" title="ROC Curve of RFC">
  <p align="center">Confusion Matrix and ROC curve of Random Forest Classifier</p>
</p>

Comparing with Logistic Regression there's less **False alarms** and there's more number of samples were classified with right class. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/83111155/130618398-590791be-77fe-4056-9807-94d802e79af1.png" title="PR Curve of RFC">
  <p align="center">PR curve of Random Forest Regression</p>
</p>

With precision score at **0.79** and recall score at **0.31** Random Forest Classifier performed well than previous one.But still to less **False alarms**. 
We will use this model to build an simple interface. 

# Building demo interface with model 
Gradio has this amazing interface to demo your model thus, using that we are able to see the model's ability to predict the proablility of the status of machine failure.

<p align="center">
  <img src="https://user-images.githubusercontent.com/83111155/130619271-0f39706b-7f2d-4b7c-b84e-c5043561fcda.png" title="Screenshot2">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/83111155/130619267-d56bdd09-3c3a-48b9-b18f-38be57d6801b.png" title="Screenshot2">
</p>

# Future Work 
- This project doesn't end here it has go even futher with identifying the different failure modes within the system.
- More **False Alarms** should be flagged with person in site to make sure that machine failure is need happening and also it should predict even before. 
