# DMBI - Data Hackathon for Social Good

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Challenge: IoT Device Classification
With the prevalence of IoT devices that is connected to the network of an organization, there is a risk of the device being attacked which may lead to the compromise of the security of the organization. We have a whitelist of device allowed to connect to the network and the challenge is to determine:
1. Classify if the device: is the device type known or unknown?
2. Given that the device type is known, we need to classify what kind of device it is (eg. sensors, lights, TV).

Related Files:
1. The main file is located at [Datathon Submission](https://github.com/randyjulian/DMBI_iot_classification/blob/master/Datathon%20Submission.ipynb "Datathon Submission")


### Data
Some of the features of the data are:
* no of ACK / PSH / RST packets sent / received, by client / server / both
* no of bytes sent / received, by client / server / both
* no of retransmitted packets   
* ratio between # of bytes sent and # of bytes received   
* session duration
* no of bytes sent by client over HTTP (min., Q1, median, avg., Q3, max, entropy, stdev)
* no of cookies (min, Q1, …)
* HTTP request-response inter arrival time (min, Q1, …)
* SSL handshake duration (min, Q1, …)
* ratio between ssl sessions and expired certificates (min, Q1, …)
* TCP packet time-to-live sent by client / server / both (min, Q1, …)   
* dominated host Alexa rank   
* server port is user / system / dynamic
* protocol is DNS/ HTTP / SSL

The data size are over 500,000 entries.

### Cleaning and Resampling of data

The data are relatively cleaned and structured, there are a few null entries so we removed observation with null entries. Further more, due to the low ratio of unknown data and the large amount of data, we resample the data to fit in a 80/20 ratio and to adjust the data to 1000 for training.

### Known and Unknown device_category

We separate the problem to tackle the known and unknown devices in progression.

![Classification Diagram Tree](iot_device_classification_diagram.png?raw=true "Classification Diagram Tree")

#### Unknown devices

IsolationForest is an algorithm used to detect anomalies by giving each observation a score depending if the algorithm consider the observation as an anomaly.

The IsolationForest algorithm is used in this problem to measure how well we recognize a device. The process is explained below:
1. We train the IsolationForest to learn each device. Hence we built 10 IsolationForest models, one for each device_category. For each model, we only feed in one particular device (eg: lights) in training.
2. So we have 10 models that know really well each of their own device, hence we can use this as a "recognition" model eg: feeding a motion_sensor data to a lights model will return us as unknown/0 and feeding a lights data will return us a known/1.
3. We ran the observation to all the 10 models to give us a vector of 0,1 (0 for unknown, 1 for known) and we sum this binary values across each observation.
4. By theory, for any observations, if they belong to any known device_category, the sum should be greater than 1. Hence, the devices whose sum is 0 will be defined as the unknown device_category.

#### Known devices

After running through the algorithm above, we will be able to differentiate the unknown and known devices. Then we used a combination of RandomForest and XGBoost to classify the known category.

Then we combine the two classifier together to create a combined classifer for known and unknown classes.

## Results
The algorithm achieved 89% accuracy and ranked 2nd in the IOT Device Track. In the overall competition, our group won overall 3rd prize for our accuracy and also our simple solution. The link for this the competition: https://www.kaggle.com/c/IoT_device_type_classification
