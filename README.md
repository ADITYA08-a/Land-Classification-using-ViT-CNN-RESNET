# Assignment 3: Multiclass Land-use Classification in Satellite Imagery
**Author:** Adithya Krishna  
**Program:** M.Tech Robotics  
**Course:** JRL7680 

**Make sure the directory structure is exactly as this**
  Train_scripts.py
  Testing_scripts.py
  util_files.py
  load_data.py
  utils.py
  A3_Dataset/
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    └── [Image Folders]
Also, A3_Dataset has to be in the same space in which the train and test scripts and the util files are placed.

## Environment Setup

1. Ensure you have Python 3.9 installed.
2. Create and activate a virtual environment (optional but recommended):
   bash
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On Linux/Mac:
   source env/bin/activate 
3. To install :
   pip install -r requirements.txt
   
**Execution Example**:
Task 1.1: Standard ResNet-18

Training (Cross-Entropy Baseline):
```python Task1_1_Train.py --data_dir "./A3_Dataset" --batch_size 32 --epochs 10 --lr 0.0001 --loss_type ce```

The model weights will be saved as a .pth file in the same directory in which the training script exists.

For Focal Loss :
```python Task1_1_Train.py --data_dir "./A3_Dataset" --batch_size 32 --epochs 10 --lr 0.0001 --loss_type focal --gamma 1.8750 --alpha 0.75```

Refer to Report for Hyperparameters 

For Test script:

```python Task1_1_Test.py --data_dir "./A3_Dataset" --model_path "Task1_1_best_focal.pth" ```

Task 1.2: Enhancing CNNs with Squeeze-and-Excitation (SE) Blocks

Training (Cross-Entropy Loss)


```python Task1_2_Train.py --data_dir "./A3_Dataset" --batch_size 32 --epochs 10 --lr 0.0001 --loss_type ce```

For Focal Loss:

```python Task1_2_Train.py --data_dir "./A3_Dataset" --batch_size 32 --epochs 10 --lr 0.0001 --loss_type focal --gamma 2.0 --alpha 0.25```

For Testing:
```python Task1_2_Test.py --data_dir "./A3_Dataset" --model_path "Task1_2_best_model_focal_loss.pth" ```

Task 2.1 :

For Training(Cross-Entropy):

```python Task2_1_Train.py --data_dir "./A3_Dataset" --batch_size 32 --epochs 10 --lr 0.0001 --loss_type ce ```

For Training(Focal Loss):

```python Task2_1_Train.py --data_dir "./A3_Dataset" --batch_size 32 --epochs 10 --lr 0.0001 --loss_type focal --gamma 2.0 --alpha 0.75 ```

For Testing:

```python Task2_1_Test.py --data_dir "./A3_Dataset" --model_path "task2_1_best_deit_focal.pth" ```

Task 2.2 Transformers without Normalization (Dynamic Tanh) :

For Training (Cross Entropy ): 
```python Task2_2_Train.py --data_dir "./A3_Dataset" --batch_size 32 --epochs 10 --lr 0.0001 --loss_type ce ```

For Training (Focal Loss) :

```python Task2_2_Train.py --data_dir "./A3_Dataset" --batch_size 32 --epochs 10 --lr 0.0001 --loss_type focal --gamma 2.0 --alpha 0.5 ```

For Testing :

```python Task2_2_Test.py --data_dir "./A3_Dataset" --model_path "Task2_2_best_focal.pth" ```


Task 3.1 : GRAD_CAM Visualization for CNNs:

``` python Task3_1_GradCAM.py --data_dir "./A3_Dataset" --resnet_weights "Task1_1_best_focal.pth" --se_resnet_weights "Task1_2_best_model_focal_loss.pth" ```

Task 3.2 : Attention Maps for DeiT-3

``` python Task3_2_Attention.py --data_dir "./A3_Dataset" --deit_weights "task2_1_best_deit_ce.pth" ```


**Note : For Cross Entropy and Focal Loss, There will be seperate model weights saved. For Testing crossentropy. The cross entopy model will have '..ce.pth' or '..cross-entropy-loss.pth' and the model from focal loss will have 'focal_loss.pth' or 'focal.pth in the name**

**------------------------**
**------------------------**
**LOCATION OF MODEL WEIGHTS**

``` The cross entropy model weights for Task1.1, 1.2, 2.1, 2.2 are available in this link ``
***https://drive.google.com/drive/folders/1EW-swhZjCLggZY1ZzISwxQ6M6nm60Q6q?usp=sharing**

``` The focal loss based model weights for Task1.1, 1.2, 2.1, 2.2 are available in this link ``

**https://drive.google.com/drive/folders/18ke5XzqWfkS27aO-gJguc1DLG8yp5VIK?usp=sharing ***