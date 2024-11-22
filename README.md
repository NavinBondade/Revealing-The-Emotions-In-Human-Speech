# Revealing The Emotions In Speech With Deep Learning
<p align="center">
<a href="https://nbviewer.jupyter.org/github/NavinBondade/Revealing-The-Emotions-In-Human-Speech/blob/main/Notebook/Recognizing_The_Emotions_in_Speech.ipynb" target="_blank">
  <img align="center"  src="https://github.com/NavinBondade/Distinguishing-Fake-And-Real-News-With-Deep-Learning/blob/main/Graphs/button_if-github-fails-to-load-the-notebook-click-here%20(4).png?raw=true"/>
</a>
</p>
<img src="https://static01.nyt.com/images/2013/10/13/business/13-TECHNO-SUB/13-TECHNO-SUB-superJumbo.jpg" width="950" height="520">
<p>HHuman communication extends beyond words; it also encompasses the nuances of speech, such as tone and pitch, which can dramatically alter the meaning of a message. Recognizing the complexity of this form of communication, I have developed a sophisticated deep-learning system designed to capture these subtle variations in human speech. This system is capable of accurately identifying and interpreting the emotions conveyed through spoken language. The model is trained to detect seven distinct emotions: Angry, Happy, Neutral, Sad, Fearful, Disgusted, and Surprised. By analyzing these emotional cues, the system provides a deeper understanding of the underlying sentiment in speech, making it a valuable tool for applications in areas such as customer service, mental health, and human-computer interaction.
</p>
<h2>Libraries Used</h2> 
<ul>
  <li>Tensorflow</li>
  <li>Keras</li>
  <li>Numpy</li>
  <li>Pandas </li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
  <li>Librosa</li>
  <li>Sklearn</li>
</ul>
<h2>Audio Analysis</h2>
<img src="https://github.com/NavinBondade/Revealing-The-Emotions-In-Human-Speech/blob/main/Graphs%20and%20Pictures/Wave%20Plot%20For%20All%207%20Different%20Types%20Of%20Emotions.png">
<br>
<img src="https://github.com/NavinBondade/Revealing-The-Emotions-In-Human-Speech/blob/main/Graphs%20and%20Pictures/Spectrogram%20For%20All%207%20Different%20Types%20Of%20Emotions.png">
<br>
<img src="https://github.com/NavinBondade/Revealing-The-Emotions-In-Human-Speech/blob/main/Graphs%20and%20Pictures/Chromagram%20For%20All%207%20Different%20Types%20Of%20Emotions.png">
<h2>Target Class Distribution</h2>
<p align="center"> 
<img src="https://github.com/NavinBondade/Revealing-The-Emotions-In-Human-Speech/blob/main/Graphs%20and%20Pictures/Target%20Class%20Distribution.png">
</p><br>
<p align="center"> 
<img src="https://github.com/NavinBondade/Revealing-The-Emotions-In-Human-Speech/blob/main/Graphs%20and%20Pictures/Target%20Distribution%20In%20Percentage.png">
</p>      
<h2>Model Details</h2>
<p>
To effectively capture the emotions embedded within speech, I employed the Mel-Frequency Cepstral Coefficients (MFCCs) technique for feature extraction from the audio data. This process laid the groundwork for the development of a deep learning model, which is structured around Long Short-Term Memory (LSTM) layers and fully connected neural layers. The model architecture comprises two LSTM layers and three dense layers. The LSTM layers, known for their ability to learn and retain long-term dependencies through their unique gating mechanisms, are crucial in processing the temporal aspects of speech data. The fully connected layers then leverage this information to accurately classify the audio into one of the predefined emotion categories.</p>
<p>Each layer in the model, except for the final dense layer, utilizes the Rectified Linear Unit (ReLU) activation function, which introduces non-linearity and helps the model learn complex patterns. The final dense layer, however, uses the softmax activation function, transforming the output into probabilities that can be interpreted as the likelihood of the audio belonging to each emotion class. To ensure the model's robustness and prevent overfitting, techniques such as batch normalization and dropout are integrated into the architecture, enhancing its generalization capabilities.<p>
<h2>Model Training</h2>   
<img src="https://github.com/NavinBondade/Revealing-The-Emotions-In-Human-Speech/blob/main/Graphs%20and%20Pictures/loss-accuracy.png">
<p>The model underwent a comprehensive training process spanning 33 epochs to optimize its performance on the given task. During each epoch, the model iteratively processed the entire training dataset, allowing it to progressively learn and refine its internal parameters for improved accuracy and generalization.

For optimization, the model employed the Stochastic Gradient Descent (SGD) algorithm with a learning rate of 0.001 and a momentum factor of 0.9. SGD is a widely used optimization technique that updates the model's weights incrementally by calculating the gradient of the loss function with respect to the weights for a subset of the data. The chosen learning rate of 0.001 dictates the step size at which the model updates its weights during training; a relatively small learning rate like this ensures stable and gradual convergence towards the minimum of the loss function, reducing the risk of overshooting optimal values.

Incorporating a momentum term of 0.9 further enhances the efficiency of the SGD optimizer. Momentum helps accelerate SGD in relevant directions and dampens oscillations by accumulating a fraction of the previous weight updates. This leads to faster convergence, especially in scenarios where the loss surface has many local minima or is ravine-shaped, by maintaining consistent progress and preventing the optimization process from getting stuck in suboptimal points.

The training process utilized the categorical cross-entropy loss function to evaluate and guide the model's learning. Categorical cross-entropy is particularly suitable for multi-class classification problems as it quantifies the difference between the predicted probability distribution and the true distribution of the classes. By penalizing the model more heavily for incorrect or confident yet wrong predictions, this loss function effectively encourages the model to output probability distributions that closely align with the actual distribution of the data. This rigorous penalization mechanism ensures that the model not only learns to make correct predictions but also accurately represents the confidence levels associated with each prediction.

Overall, the combination of multiple training epochs, the SGD optimizer with carefully chosen learning rate and momentum, and the categorical cross-entropy loss function collectively contributed to a robust and effective training regimen. This setup enabled the model to systematically reduce prediction errors and enhance its ability to generalize from the training data to unseen data, thereby achieving reliable and accurate performance in its designated tasks.</p>
<h2>Model Evaluation</h2>
<p>The model demonstrates strong performance, achieving a training accuracy of 92% with a corresponding loss of 0.22, indicating it has effectively learned the patterns in the training data. Moreover, its test accuracy of 94% and reduced loss of 0.12 suggest that the model generalizes well to unseen data, performing even better on the test set. The lower loss on the test data, combined with the higher accuracy, underscores the model's robustness and ability to make precise predictions without overfitting, making it well-suited for real-world applications.</p>
<ul>
  <li><b>Training Data Accuracy: 92%</b></li>
  <li><b>Training Data Loss: 0.22</b></li>
  <li><b>Test Data Accuracy: 94%</b></li> 
  <li><b>Test Data Loss: 0.12</b></li> 
</ul>
<h2>Dataset</h2>
<p>The dataset on which model trained is from University of Toronto, Psychology Department and can be downloaded from kaggle: https://bit.ly/3yAatHM or from the official website: https://bit.ly/3yzKv7B.</p>
<h2>Conclusion</h2>
<p>In this project, I have created an LSTM based deep learning system that is capable of determining seven emotions: Angry, Happy, Neutral, Sad, Fearful, Disgusted, and Suprised in human speech with an impressive accuracy of 94 percent.</p>
                                                                                                                                                      
