# Federated Unlearning
This code is a pipeline to train and unlearn a CNN in a federated and centralised fashion on MNIST. 
### Centralised (Un-)learning
First we started training a centralised model on MNIST and unlearned it via a projected gradient ascent to later on then relearn the model with only part of the data. This was used as a baseline and goal for the federated experiments. 
### Federated (Un-)learning
For te training in the federated fashion, we tried different training splits of 3, 5 and 10 clients. The goal of the unlearning process was to completely remove one clients data and only use the remaining clients data to retrain the model.
### Data heterogeneity
Data in federated learning is intrinsically more heterogenous, as the different clients or parties are usually located at different points or assembled data slightly differently causing inconsitencies. Thus it was crucial to also test the data on different data distributions among the clients.
### Additional Information
For a more detailed explanation please read the pdf Federated_Unlearning_For_FeatureCloud.pdf in the repo. <br />
The unlearning algorithm was also added to the FeatureCloud platform: https://github.com/AntoniaGocke/dumplings <br />
Contributors to this project were: Caterine Roncalli, Antonia Gocke, Kester Bagemihl and Simon Feldmann
