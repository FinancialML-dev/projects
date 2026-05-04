"""
Stock/Crypto Price Predictor 

Predicts next candle direction (up/down) using historical price data.
--> Uses simple neural network trained on price returns. 

Author: Leo
Date: 2026-04-28
"""

import torch                                #uv add torch 
import torch.nn 
import numpy 
import matplotlib.pyplot as plotLibrary     #uv add matplotlib 
from datetime import datetime 
from timebarsRefactoredFull import fetchCryptoData, formatDataToLists 
from alpaca.data.timeframe import TimeFrame 

#-------------------------------
#Configuration 
#-------------------------------
CONFIG = {
    "SYMBOL": "BTC/USD",
    "TIMEFRAME": TimeFrame.Minute,          #Intervall in Minutes  
    "START_DATE": datetime(2026, 2, 1),     #Start date year, month, day --> of minute data 
    "END_DATE": datetime(2026, 4, 1),       #End date year, month, day --> of minute data
    "LOOKBACK": 32,                         #Use last <30> candles for prediction 
    "TRAIN_SPLIT": 0.8,                     #80% training data, 20% test data 
    "EPOCHS": 500,                          #<100> Iterations
    "LEARNING_RATE": 0.001,                 #0.001 
    "BATCH_SIZE": 32
}

#-------------------------------
#Feature Engineering (input features, variables)
#-------------------------------
def calculateReturns(_prices): 
    """
    Calculates the percentage return between each consecutive price.
    (Calculate price returns (percent change)).

    Args:
        _prices (list): List of close prices.

    Returns:
        numpy.ndarray: Array of returns with length N-1 (one less than input).

    Example:
        >>> returns = calculateReturns([100, 105, 98, 110])
        >>> print(returns)
        [ 0.05  -0.0667  0.1224]   #+5%, -6.67%, +12.24%
    """

    prices = numpy.array(_prices)
    returns = (prices[1:] - prices[:-1]) / prices[:-1]      #This calculates the percentage return between consecutive prices.
    
    return returns 

def createFeaturesAndLabels(_closePrices, _lookback=30): 
    """
    Builds a sliding-window dataset of return sequences and binary direction labels.
    (Create feature windows and labels for supervised learning.)
    
    For each position i, the feature is a window of the previous `_lookback` returns,
    and the label is 1 if the next return is positive (price up), 0 if negative (price down).
    (For each time step t, features are returns from t-lookback to t-1.
    Label is whether price went up at time t --> (1) or down (0).)

    Args:
        _closePrices (list): List of close prices.
        _lookback (int, optional): Number of past returns (candles) to use as features. Defaults to 30.

    Returns:
        tuple:
            - features (numpy.ndarray): Shape (N, _lookback) — each row is a return window.
            - labels (numpy.ndarray): Shape (N,) — 1 for price up, 0 for price down.

    Example:
        >>> prices = [100, 102, 101, 105, 103, 107]  # 6 prices → 5 returns
        >>> features, labels = createFeaturesAndLabels(prices, _lookback=3)
        >>> print(features.shape)   # (2, 3)  — 2 windows of 3 returns each
        >>> print(labels)           # [1, 0]  — next move up, then down
    """

    returns = calculateReturns(_closePrices)
    
    features = []
    labels = []
    
    for i in range(_lookback, len(returns)):
        #Features (inputs): returns from [i-_lookback:i-1]  
        window = returns[i-_lookback:i]
        features.append(window)
        
        #Label: 1 if next return positive, 0 if negative 
        nextReturn = returns[i] 
        labels.append(1 if nextReturn > 0 else 0) 
    
    return numpy.array(features), numpy.array(labels)

def trainTestSplit(_X, _y, _trainRatio=0.8): 
    """
    --> Split data into training and testing sets.
    
    Splits features (_X -> inputs) and labels (_y -> outputs) into chronological train and test sets.

    Splits at a fixed index (no shuffling) to preserve time order —
    training on past data, testing on future data.

    Args:
        _X (numpy.ndarray): Feature array of shape (N, lookback).
        _y (numpy.ndarray): Label array of shape (N,).
        _trainRatio (float, optional): Fraction of data used for training. Defaults to 0.8.

    Returns:
        tuple: (XtrainData, XtestData, yTrainData, yTestData)
            - XtrainData: First 80% of features (training).
            - XtestData:  Last  20% of features (testing).
            - yTrainData: First 80% of labels   (training).
            - yTestData:  Last  20% of labels   (testing).

    Example:
        >>> features, labels = createFeaturesAndLabels(closePrices, _lookback=30)
        >>> XtrainData, XtestData, yTrainData, yTestData = trainTestSplit(features, labels, _trainRatio=0.8)
        >>> print(XtrainData.shape)   # (N*0.8, 30)
        >>> print(XtestData.shape)    # (N*0.2, 30)
    """
    splitIndex = int(_trainRatio * len(_X))     #--> 0.8 * len(features) 
    
    XtrainData = _X[:splitIndex]                #all the first values to the splitIndex (80% of the first values)
    XtestData = _X[splitIndex:]                 #all the last values starting from the splitIndex (20% of the last values)
    yTrainData = _y[:splitIndex]                #... (80% of the first values)
    yTestData = _y[splitIndex:]                 #... (20% of the last values)
    
    return XtrainData, XtestData, yTrainData, yTestData 

#-------------------------------
#Model definition (Neural Network)
#-------------------------------
class PricePredictor(torch.nn.Module): 
    """
    Binary classifier neural network that predicts price direction (up/down).

    Architecture:
        Input(_inputSize) → Linear(64) → ReLU → Dropout(0.2)
                          → Linear(32) → ReLU → Dropout(0.2)
                          → Linear(1)  → Sigmoid → output (0.0 - 1.0)

    The output is a probability — values above 0.5 indicate price going up (1),
    values below 0.5 indicate price going down (0).

    Args:
        _inputSize (int): Number of input features (equal to the lookback window size).

    Example:
        >>> model = PricePredictor(_inputSize=30)       # 30 = lookback window
        >>> features, labels = createFeaturesAndLabels(closePrices, _lookback=30)
        >>> inputTensor = torch.tensor(features[0], dtype=torch.float32)
        >>> prediction = model(inputTensor)
        >>> print(prediction)
        tensor([0.73])                                  # 73% chance price goes up
    """
    def __init__(self, _inputSize):
        super().__init__()
        self.neuralNetwork = torch.nn.Sequential(   
            torch.nn.Linear(_inputSize, 64),            #input Layer (layer 1): _inputSize (features) --> Hidden layer: 64 neurons/nodes
            torch.nn.ReLU(),                            #Activation function: rectified linear unit (ReLU) applied element-wise to each hidden neuron
            torch.nn.Dropout(0.2),                      #randomly turns off 20% of neurons during each training. --> Why? To prevent overfitting. 
            torch.nn.Linear(64, 32),                    #Hidden Layer (layer 2): 64 neurons/nodes --> Second Hidden layer: 32 neurons/nodes
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),                     #Second Hidden Layer (layer 3): 32 neurons/nodes --> Output layer (target): 1 neuron/node (1 output) 
            torch.nn.Sigmoid()                          #Squashes any number into a value between 0 and 1. Express the answer as a probability
        )
        
    def forward(self, _x):
        return self.neuralNetwork(_x)

#-------------------------------
#Training: Train the model (our Neural Network)
#-------------------------------
def trainModel(_model, _Xtrain, _yTrain, _epochs=100, _learningRate=0.001, _batchSize=32): 
    """
    Trains the PricePredictor model using binary cross-entropy loss and Adam optimizer.

    Runs a forward pass, computes loss, backpropagates gradients, and updates
    weights each epoch. Prints loss and accuracy every 20 epochs.

    Args:
        _model (PricePredictor): The neural network to train.
        _Xtrain (numpy.ndarray): Training features of shape (N, lookback).
        _yTrain (numpy.ndarray): Training labels of shape (N,) with values 0 or 1.
        _epochs (int, optional): Number of training iterations. Defaults to 100.
        _learningRate (float, optional): Step size for Adam optimizer. Defaults to 0.001.
        _batchSize (int, optional): Batch size (reserved for future use). Defaults to 32.

    Returns:
        dict: Training history with keys:
            - "loss"     (list): Loss value recorded each epoch.
            - "accuracy" (list): Accuracy value recorded each epoch.

    Example:
        >>> model = PricePredictor(_inputSize=30)
        >>> XtrainData, XtestData, yTrainData, yTestData = trainTestSplit(features, labels)
        >>> history = trainModel(model, XtrainData, yTrainData, _epochs=100, _learningRate=0.001)
        --------------------
        Training...
        --------------------
        Epoch/Iteration   0: Loss=0.7124, Accuracy=0.4823
        Epoch/Iteration  20: Loss=0.6891, Accuracy=0.5241
        Epoch/Iteration  40: Loss=0.6543, Accuracy=0.5780
        ...
        --------------------
    """
    XtrainData = torch.FloatTensor(_Xtrain)
    yTrainData = torch.FloatTensor(_yTrain).reshape(-1, 1) 
    
    optimizer = torch.optim.Adam(_model.parameters(), lr=_learningRate)
    criterion = torch.nn.BCELoss()
    
    history = {"loss": [], "accuracy": []}
    
    print("\n" + "-"*20)
    print("Training...")
    print("-"*20)
    
    for item in range(_epochs):
        _model.train()
        
        #Forward pass
        predictions = _model(XtrainData)
        loss = criterion(predictions, yTrainData)
        
        #Backward pass
        optimizer.zero_grad()       #Reset/Clear all gradients 
        loss.backward()             #partial derivative of dLoss/dmodel = ..., dLoss/dy = ... [chain rule applied]
        optimizer.step()            #Performs a single optimization step
        
        #Calculate accuracy 
        accuracy = ( (predictions > 0.5).float() == yTrainData ).float().mean()
        
        history["loss"].append(loss.item())
        history["accuracy"].append(accuracy.item())
        
        if item % 20 == 0:
            print(f"Epoch/Iteration {item:3d}: Loss={loss.item():.4f}, Accuracy={accuracy.item():.4f} ")
            #print(f"Iteration {item:3d}: loss={loss.item():.4f}")  
            print(f"    Layer 1: w={_model.neuralNetwork[0].weight.data.shape}, b={_model.neuralNetwork[0].bias.data.shape}") #Our model wraps the Sequential inside `self.neuralNetwork` --> So, For Sequential model we use: model.neuralNetwork[0] the first Linear(x, y) layer. Layer 1 --> shape: x rows, y column 
            print(f"    Layer 2: w={_model.neuralNetwork[3].weight.data.shape}, b={_model.neuralNetwork[3].bias.data.shape}") #model.neuralNetwork[3] the second Linear(x, y) layer. Layer 2 --> shape: x row, y columns  
            print(f"    Layer 3: w={_model.neuralNetwork[6].weight.data.shape}, b={_model.neuralNetwork[6].bias.data.shape}") #model.neuralNetwork[6] the second Linear(x, y) layer. Layer 3 --> shape: x row, y columns  
    
    print("-"*20 + "\n")
    
    return history 


#-------------------------------
#Evaluation 
#-------------------------------
def evaluateModel(_model, _Xtest, _yTest): 
    """
    Evaluate model on test set.
    Evaluates the trained model on unseen test data and returns classification metrics.

    Runs inference with Dropout disabled (model.eval() + no_grad), then computes
    accuracy, baseline, precision, recall, F1, and a confusion matrix.

    Args:
        _model (PricePredictor): The trained neural network.
        _Xtest (numpy.ndarray): Test features of shape (N, lookback).
        _yTest (numpy.ndarray): Test labels of shape (N,) with values 0 or 1.

    Returns:
        dict: Evaluation metrics with keys:
            - "accuracy"        (float): % of correct predictions.
            - "baseline"        (float): % accuracy of always guessing the most common class.
            - "precision"       (float): Of all predicted UP, how many were actually UP.
            - "recall"          (float): Of all actual UP, how many were correctly predicted.
            - "f1"              (float): Harmonic mean of precision and recall.
            - "confusionMatrix" (dict):  truePositive, falsePositive, trueNegative, falseNegative.

    Example:
        >>> history = trainModel(model, XtrainData, yTrainData, _epochs=100)
        >>> metrics = evaluateModel(model, XtestData, yTestData)
        >>> print(metrics["accuracy"])   # 0.58  — model correct 58% of the time
        >>> print(metrics["baseline"])   # 0.53  — naive guess correct 53% of the time
        >>> print(metrics["f1"])         # 0.61  — balanced precision/recall score
    """
    _model.eval() 
    
    with torch.no_grad():
        XtestData = torch.FloatTensor(_Xtest)
        yTestData = torch.FloatTensor(_yTest).reshape(-1, 1)   #The `-1` means "figure this dimension out yourself." Telling NumPy/PyTorch: "I know I want 1 column — you calculate how many rows are needed."
        
        #Predictions
        predictions = _model(XtestData)
        predictedClass = (predictions > 0.5).float() 
        
        #Metrics 
        accuracy = (predictedClass == yTestData).float().mean().item()
        
        #Baseline: always predict most common class 
        baseline = max(_yTest.mean(), 1 - _yTest.mean()) 
        
        #True positives, false positives, etc.
        truePositive = ( (predictedClass == 1) & (yTestData == 1) ).sum().item()
        falsePositive = ( (predictedClass == 1) & (yTestData == 0) ).sum().item()
        trueNegative = ( (predictedClass == 0) & (yTestData == 0) ).sum().item()
        falseNegative = ( (predictedClass == 0) & (yTestData == 1) ).sum().item()

        precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0
        recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) > 0 else 0
        f1 = 2*(precision*recall) / (precision + recall) if (precision + recall) > 0 else 0 

    
    return {
        "accuracy": accuracy,
        "baseline": baseline,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusionMatrix": {
            "truePositive": truePositive, 
            "falsePositive": falsePositive,
            "trueNegative": trueNegative,
            "falseNegative": falseNegative
        }
    }

def printEvaluation(_metrics): 
    """
    Prints evaluation metrics from evaluateModel() in a formatted report (readable format).

    Displays accuracy, baseline comparison, improvement over baseline,
    precision, recall, F1 score, and a confusion matrix.

    Args:
        _metrics (dict): Output from evaluateModel() containing keys:
            "accuracy", "baseline", "precision", "recall", "f1", "confusionMatrix".

    Returns:
        None

    Example:
        >>> metrics = evaluateModel(model, XtestData, yTestData)
        >>> printEvaluation(metrics)
        xxxxxxxxxxxxxxxxxxxx
        Evaluation results
        xxxxxxxxxxxxxxxxxxxx
        Test Accuracy:          0.5800
        Baseline (majority):    0.5300
        Improvement:            0.0500

        Precision:              0.6100
        Recall:                 0.5500
        F1 Score:               0.5783

        Confusion Matrix:
            True Positive:  45      False Positive: 29
            False Negative: 37      True Negative:  51
        xxxxxxxxxxxxxxxxxxxx
    """
    print("\n" + "x"*20)
    print("Evaluation results")
    print("x"*20)
    print(f"Test Accuracy:          {_metrics["accuracy"]:.4f}")
    print(f"Baseline (majority):    {_metrics["baseline"]:.4f}")
    print(f"Improvement:            {(_metrics["accuracy"] - _metrics["baseline"]):.4f}")
    print(f"\nPrecision:            {_metrics["precision"]:.4f}")
    print(f"Recall:                 {_metrics["recall"]:.4f}")
    print(f"F1 Score:               {_metrics["f1"]:.4f}")
    
    confusionMatrix = _metrics["confusionMatrix"]
    print(f"\nConfusion Matrix: ")
    print(f"    True Positive:  {confusionMatrix["truePositive"]:<6}  False Positive: {confusionMatrix["falsePositive"]}")
    print(f"    False Negative: {confusionMatrix["falseNegative"]:<6} True Negative:  {confusionMatrix["trueNegative"]}")
    print("x"*20 + "\n")
    
    
#-------------------------------
#Visualization 
#-------------------------------
def plotTrainingHistory(_history): 
    """
    Plots training loss and accuracy over epochs/iterations as a side-by-side chart.

    Left panel  — loss curve (should decrease over time).
    Right panel — accuracy curve with a red dashed line at 0.5 (random guess baseline).
    Saves the chart to "Training-history.png" and displays it.

    Args:
        _history (dict): Output from trainModel() with keys:
            - "loss"     (list): Loss value per epoch/iteration.
            - "accuracy" (list): Accuracy value per epoch/iteration.

    Returns:
        None

    Example:
        >>> history = trainModel(model, XtrainData, yTrainData, _epochs=100)
        >>> plotTrainingHistory(history)
        --> Training history plot saved to Training-history.png
    """
    figure, (axis1, axis2) = plotLibrary.subplots(1, 2, figsize=(12, 4))
    
    #Loss plot 
    axis1.plot(_history["loss"])
    axis1.set_title("Training loss")
    axis1.set_xlabel("Epoch/iteration")
    axis1.set_ylabel("BTC Loss")
    axis1.grid(True, alpha=0.3)
    
    #Accuracy plot 
    axis2.plot(_history["accuracy"])
    axis2.set_title("Training Accuracy")
    axis2.set_xlabel("Epoch/iteration")
    axis2.set_ylabel("Accuracy")
    axis2.axhline(y=0.5, color="red", linestyle="--", label="Random guess")
    axis2.legend()
    axis2.grid(True, alpha=0.3)
    
    plotLibrary.tight_layout()
    plotLibrary.savefig("Training-history.png", dpi=150, bbox_inches="tight")
    print("--> Training history plot saved to training-history.png")
    plotLibrary.show()

#-------------------------------
#Main Execution 
#-------------------------------    
def main():
    """
    Main execution pipeline. 
    """
    print("\n" + "-"*20)
    print("Crypto price predictor")
    print("-"*20 + "\n")
    
    #1. Fetch data 
    print("Step 1: Fetching data...")
    bars = fetchCryptoData(
        CONFIG["SYMBOL"],
        CONFIG["TIMEFRAME"],
        CONFIG["START_DATE"],
        CONFIG["END_DATE"]
    )
    
    time, openPrice, highPrice, lowPrice, closePrice, volumeUnits = formatDataToLists(bars, CONFIG["SYMBOL"]) #Version 1
    #_, _, _, _, close, _ = formatDataToLists(bars, CONFIG["SYMBOL"]) #Version 2
    
    #2. Feature engineering 
    print("\nStep 2: Creating features...")
    X, y = createFeaturesAndLabels(closePrice, _lookback=CONFIG["LOOKBACK"])
    print(f"--> Features shape: {X.shape}")
    print(f"--> Labels shape: {y.shape}")
    print(f"--> Positive class ratio: {y.mean():.2%}")

    #3. Train/Test split 
    print("\nStep 3: Splitting data...")
    XtrainData, XtestData, yTrainData, yTestData = trainTestSplit(
        X, y, _trainRatio=CONFIG["TRAIN_SPLIT"] 
    )
    print(f"--> Train samples: {len(XtrainData)}")
    print(f"--> Test samples:  {len(XtestData)}")
    
    #4. Build model 
    print("\nStep 4: Building model...")
    torch.manual_seed(42) #Remove or comment out, when you want variance across multiple runs (for the `runMultipleTimes(_iterations)` ).
    model = PricePredictor(_inputSize=X.shape[1])
    totalParameters = sum(_parameter.numel() for _parameter in model.parameters())
    print(f"--> Model created with {totalParameters} parameters")
    
    #5. Train 
    print("\nStep 5: Training model...")
    history = trainModel(model, XtrainData, yTrainData, 
                         _epochs=CONFIG["EPOCHS"], 
                         _learningRate=CONFIG["LEARNING_RATE"])
    
    #6. Evaluate 
    print("\nStep 6: Evaluating the model...")
    metrics = evaluateModel(model, XtestData, yTestData)
    printEvaluation(metrics)
    
    #7. Visualize 
    print("\nStep 7: Generating plots...")
    plotTrainingHistory(history)
    
    #8. Save model 
    print("\nStep 8: Saving the model...")
    torch.save(model.state_dict(), "pricePredictor.pth")
    print("--> Model saved to pricePredictor.pth\n")
    
    print("-"*20)
    print("Pipeline completed!")
    print("-"*20 + "\n")
     
    
if __name__ == "__main__":
    main()
      

#-----
#Small manual tests, for each of the function examples and class examples   
#-----
'''returns = calculateReturns([100, 105, 98, 110])
print(returns)'''

'''prices = [100, 102, 101, 105, 103, 107]  # 6 prices → 5 returns
features, labels = createFeaturesAndLabels(prices, _lookback=3)
print(features.shape)   # (2, 3)  — 2 windows of 3 returns each
print(labels)           # [1, 0]  — next move up, then down '''

'''closePrices = [100, 102, 101, 105, 103, 107]  # 6 prices → 5 returns
features, labels = createFeaturesAndLabels(closePrices, _lookback=30)
XtrainData, XtestData, yTrainData, yTestData = trainTestSplit(features, labels, _trainRatio=0.8)
print(XtrainData.shape)   # (N*0.8, 30)
print(XtestData.shape)    # (N*0.2, 30)'''


'''closePrices = [100, 102, 101, 105, 103, 107, 87, 99, 120, 103, 76, 83, 79, 90, 92, 95, 89, 99, 111, 103, 107, 89, 99, 102, 84, 85, 87, 108, 103, 121, 119, 117]  # 32 prices → 31 returns. Explaination: 30 (lookback) + 1 (label) = 31 returns, so we need 32 prices. "What happened next?" this is the prediction... 
model = PricePredictor(_inputSize=30)       # 30 = lookback window
features, labels = createFeaturesAndLabels(closePrices, _lookback=30)
inputTensor = torch.tensor(features[0], dtype=torch.float32)
prediction = model(inputTensor)
print(prediction)'''

'''closePrices = [100, 102, 101, 105, 103, 107, 87, 99, 120, 103, 76, 83, 79, 90, 92, 95, 89, 99, 111, 103, 107, 89, 99, 102, 84, 85, 87, 108, 103, 121, 119, 117]  # 32 prices → 31 returns. Explaination: 30 (lookback) + 1 (label) = 31 returns, so we need 32 prices. "What happened next?" this is the prediction... 
model = PricePredictor(_inputSize=30)       # 30 = lookback window
features, labels = createFeaturesAndLabels(closePrices, _lookback=30)
inputTensor = torch.tensor(features[0], dtype=torch.float32)
prediction = model(inputTensor)
print(prediction)'''

'''closePrices = [100, 102, 101, 105, 103, 107, 87, 99, 120, 103, 76, 83, 79, 90, 92, 95, 89, 99, 111, 103, 107, 89, 99, 102, 84, 85, 87, 108, 103, 121, 119, 117]  # 32 prices → 31 returns. Explaination: 30 (lookback) + 1 (label) = 31 returns, so we need 32 prices. "What happened next?" this is the prediction... 
X, y = createFeaturesAndLabels(closePrices, _lookback=CONFIG["LOOKBACK"])

XtrainData, XtestData, yTrainData, yTestData = trainTestSplit(
        X, y, _trainRatio=CONFIG["TRAIN_SPLIT"] 
    )

model = PricePredictor(_inputSize=X.shape[1])

history = trainModel(model, XtrainData, yTrainData, _epochs=100) #_epochs=1 Just for testing so it runs the example code. Note! you need more epochs/iterations to get good values...
metrics = evaluateModel(model, XtestData, yTestData)
print(metrics["accuracy"])   # 0.58  — model correct 58% of the time
print(metrics["baseline"])   # 0.53  — naive guess correct 53% of the time
print(metrics["f1"])         # 0.61  — balanced precision/recall score '''

'''metrics = evaluateModel(model, XtestData, yTestData)
printEvaluation(metrics)'''

'''history = trainModel(model, XtrainData, yTrainData, _epochs=100)
plotTrainingHistory(history)'''
