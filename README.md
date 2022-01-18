# PATRL

This repo contains code for 'PATRL': Translating Intrusion Alerts to Cyberattack Stagesusing Pseudo-Active Transfer Learning
This code was used to produce the results for our paper in the proceedings of IEEE Conference of Communcations and Network Security (CSN) 2021.

Abstract:
Intrusion alerts continue to grow in volume, variety, and complexity. 
Its cryptic nature requires substantial time and expertise to interpret the intended consequence of observed malicious actions.
To assist security analysts in effectively diagnosing what alerts mean, this work develops a novel machine learning approach that translates alert descriptions to intuitively interpretable Action-Intent-Stages (AIS) with only 1\% labeled data. 
We combine transfer learning, active learning, and pseudo labels and develop the Pseudo-Active Transfer Learning (PATRL) process.
The PATRL process begins with an unsupervised-trained language model using MITRE ATT\&CK, CVE, and IDS alert descriptions. 
The language model feeds to an LSTM classifier to train with 1\% labeled data and is further enhanced with active learning using pseudo labels predicted by the iteratively improved models.
Our results suggest PATRL can predict correctly for 85\% (top-1 label) and 99\% (top-3 labels) of the remaining 99\% unknown data.
Recognizing the need to build confidence for the analysts to use the model, the system provides Monte-Carlo Dropout Uncertainty and Pseudo-Label Convergence Score for each of the predicted alerts. 
These metrics give the analyst insights to determine whether to directly trust the top-1 or top-3 predictions and whether additional pseudo labels are needed. 
Our approach overcomes a rarely tackled research problem where minimal amounts of labeled data do not reflect the truly unlabeled data's characteristics. 
Combining the advantages of transfer learning, active learning, and pseudo labels, the PATRL process translates the complex intrusion alert description for the analysts with confidence.