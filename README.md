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
                                                                                                                                                      
