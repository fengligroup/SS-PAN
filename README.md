# SS-PAN
SS-PAN is a stacking learning framework to predict η for AN-based ceramics. In SS-PAN, a comprehensive array of physical features pertaining to both A-/B-site elements are extracted and then selected by using MDI methods combined with exhaustive search, then input into a two-layer stacking model for prediction.



Email: [fengli@ahu.edu.cn](mailto:fengli@ahu.edu.cn)

The repo is organised as follows:
- 'requirements.txt': environment file with all dependencies needed to run the project

- 'MeanDecreaseImpurity.py': Python script of the MDI method for feature selection.

- 'ExhaustiveSearch.py': Python script of the exhaustive search method for feature selection.

- 'Stacking.py': Python script for stacking model construction and evaluation.
  
- 'Predict.py':  Python script for predicting on latent space.

- '/data':
  - 'dataset_example.xlsx':  Training dataset example for SS-PAN;
  - 'latentSpace.xlsx':   LatentSpace to predict;   


# Installation
- Requirement
  OS：
  
  - Windows ：Windows7 or later
    
  Python：

  - Python >= 3.11

- Download
  
  ```bash
  git clone https://github.com//fengligroup//SS-PAN 
  ```
- Install requirement
  
  ```
  pip install -r requirement.txt
  ```

# Usage
- Data collection:
  Collect and organize the data following the format of 'dataset_example.xlsx'.

- Feature selection:
  Use 'MeanDecreaseImpurity.py' and 'ExhaustiveSearch.py' in sequence for feature selection.

- Model construction and predict:
  Use 'Stacking.py' to train  model, then use 'Predict'.py to output the final prediction on latent space.
