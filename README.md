# dmbi-datathon

Challenge: IoT Device Classification
1. With the prevalence of IoT devices that is connected to the network of an organization, there is a risk of the device being attacked which may lead to the compromise of the security of the organization. We have a whitelist of device allowed to connect to the network and the challenge is to determine:
2. Classify if the device: is the device type known or unknown?
3. Given that the device type is known, we need to classify what kind of device it is (eg. sensors, lights, TV).

What we did:
1. We did a resampling of the data to ensure balance dataset.
2. We build a classifier for the known classes using RandomForest and XGBoost
3. We build a classifier for the unknown classes using Neural Network and Isolation Forest
4. Then we combine the two classifier together to create a combined classifer for known and unknown classes.
