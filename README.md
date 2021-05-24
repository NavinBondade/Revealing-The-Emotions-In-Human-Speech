# Revealing The Emotions In Speech With Deep Learning
<img src="https://static01.nyt.com/images/2013/10/13/business/13-TECHNO-SUB/13-TECHNO-SUB-superJumbo.jpg" width="950" height="520">
<p>Humans tend to convey the messages through speech but not just using the spoken word but also with how the message has been delivered. The same message spoken with a slightly different change in tone and the pitch can convey a very different meaning than previous one. In this project, I have created a deep learning system that is capable of capturing these slight changes in human speech and determining what emotions have been conveyed through the speech. The model has skills of identifying seven emotions in a human speech which are Angry, Happy, Neutral, Sad, Fearful, Disgusted, and Suprised.</p>
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
<p>For capturing emotions hidden within the speech, I have first used the MFCCS method for feature extraction from the audio data. Then I have created a deep learning model that utilizes the LSTM and fully connected neuron layers at its core. The model consists of two LSTM layers and three dense layers. The LSTM layers help the model to learn and remember the long-term dependencies because of its three gate mechanics and the fully connected neuron layers perform the job of correctly predicting to which class of emotion does the audio belongs. </p>
<p>All the layers use RELU as an activation function except the last dense layer that uses the softmax activation function that transforms the previous layer output between 0 and 1, so that they can be interpreted as probabilities. For regularization and avoiding overfitting of the model batch normalization and dropout layers are used in the model.<p>
<h2>Model Training</h2>   
<img src="https://github.com/NavinBondade/Revealing-The-Emotions-In-Human-Speech/blob/main/Graphs%20and%20Pictures/loss-accuracy.png">
<p>The model was get trained for 33 epochs. During training, the model uses Stochastic Gradient Descent as an optimizer with a learning rate of 0.001 and momentum of 0.9 The model uses categorical cross-entropy as the loss function to penalize the model more when it makes a false prediction.</p>
<h2>Model Evaluation</h2>
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
                                                                                                                                                      
