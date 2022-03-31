# Pancreatic Cancer Registration
Repository for registration between CT and CBCT images for patients with pancreatic cancer.

The repository includes three part: pre-processing, deep-network and evaluation. 
(pre-processing and evaluation will be updated soon)


## Deep Network ##
sample data is located under ./data. Go to deep_network
```
cd deep_network
```

### Training

A general script to train a network:
```
python r21_train.py
```
By default, this would use ./data_list/sample_train.txt and ./data_list/sample_test.txt for training and validation 

To specify training and validation list, use:
```
python r21_train --train_list /path/to/your/trainining_list --test_list /path/to/your/validation_list
```
If you have an additional test list and want to see how the model works, use: (In this case, in training list, the first 75% of the data is for trainining and the last 25% is for validation) 
```
python r21_train --train_list /path/to/your/training_list --test_list /path/to/your/test_list --use_val
```

To view more option, use:
```
python r21_train.py --help
```

### Testing
A general script to train a network:
```
python r21_test.py --saved_model ./tmp_models/sample_model/best_eval.pth.tar
```
This use the uploaded_model for testing and the test_list is ./data_list/sample_test.txt. "Saved_model" is required to a trained model.

To specify testing model 
```
python r21_test.py --saved_model ./tmp_models/sample_model/best_eval.pth.tar --test_list /path/to/your/test_list
```

