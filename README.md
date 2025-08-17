# Detector


## Output
![officeeee](https://github.com/user-attachments/assets/318e5536-11b8-4e56-acd2-33bc7fce2d26)

## Dataset Format
```
dataset:
    images/
        train
        val
    labels/
        train
        val
```

## Training Instructions
### Follow the steps below to train the model:
1) Clone the Repository
```
git clone git@github.com:n33-raj/Detector.git
cd Detector
```
2) Prepare Dataset
   - Place your dataset folder in the parent directory of this repository.
   - Update the dataset path on line 96 of ```train.py``` with the correct path to your dataset.
3) Configure Training Parameters
   - Open train.py
   - Modify the following parameters as needed:
       - Batch size
       - Number of workers
       - Epochs
4) Start the training to train the model 
```
python train.py --train
```
