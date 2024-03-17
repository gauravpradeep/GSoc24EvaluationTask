# GSoc24EvaluationTask

The project can be cloned and setup as any other Julia project. The run.jl file may be executed to train and evaluate the model on the given dataset.

# Final Submission Details

To address the task of classifying the data points as either signal or background, a deep neural network is implemented with a softmax output layer of 2 neurons. The activation function chosen for the hidden layers is relu, arrived at after trying out seeveral others. The relu activation function effectively avoids common issues such as the vanishing gradient problem and allows modeling non-linear data easily. A softmax activation at the output layer generates a probability distribution, the maximum of which is chosen as the class predicted. The crossentropy loss function was used for training, via the logitcrossentropy function in flux for its increased numerical stability. Loss was monitored throughout trianing to understand how the model is learning and suitable changes to the architecture were incorporated.\
A simple accuracy metric, the number of correct classifications, was chosen as the metric to initially evaluate the model. This evaluation is carried out throughout training for the train and val sets, and for the test set at the end of training.
