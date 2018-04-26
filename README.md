# EMG-Neural-Network

This is a Backpropagation Neural Network that reconstructs physical motion using electromyography (EMG) data. It is only trained to translate the EMG signal generated by the bicep when the arm is flexed and extended. The content of the folders are:

1. aruco_vision: Code to decode aruco markers and extract arm motion data when the arm is flexed and extended
2. data_processing: Code that processes and clean the EMG signal
3. neural_network_modeling: Neural network training and testing code
4. trained_model: Weights of the trained neural network
