# MAG-attack_anomaly_detection
Repeat findings from article "Attack and anomaly detection in IoT sensors in IoT sites using machine learning approaches"

## To run
First install all dependencies
```bash
pip install requirements.txt
```

To get results and plots
```bash
python3 main.py
```
.csv files and .png images of plots will be saved to folder results/ once program finishes.

## Details
To specify which algorithms to use and which dataset sizes, edit these two lines in main.py:

```python
selectedSizes = [0.2, 0.4]
selectedAlgorithms = ['dt', 'rf']
```
By setting them both to None, like this:
```python
selectedSizes = None
selectedAlgorithms = None
```
default values will be used:
```python
selectedSizes = [0.2, 0.4, 0.6, 0.8, 1]
selectedAlgorithms = ['logReg', 'svm', 'dt', 'rf', 'ann']
```
