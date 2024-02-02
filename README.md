# EEG-Based Seizure Detection and Analysis

![image info](./assets/Banner_EEG.jpg)

# Project Overview
This project focuses on analyzing EEG (Electroencephalogram) data to classify and detect seizure activities. Utilizing a dataset from the UCI Machine Learning Repository, we apply machine learning and data science methodologies to predict seizure attacks, potentially aiding in early intervention and patient care.


# Table of Contents
- [Usage](#usage)
- [Data Description](#data-description)
- [Methodology](#methodology)
- [Results](#results)
- [Business Applications](#potential-business-and-practical-applications)
- [Next Steps](#next-steps)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

# Usage
- You can easily clone the project and see the main jupyter file on your local machine by using following command:

```bash
# Cloning
git clone git@github.com:mrpintime/Epileptic-Seizure-Recognition.git
```
or

```bash 
git clone https://github.com/mrpintime/Epileptic-Seizure-Recognition.git
```

# Data Description
The dataset comprises EEG recordings from 500 individuals, originally stored in 5 folders with 100 files each. Each file represents the brain activity of a person over 23.6 seconds, sampled into 4097 data points. We transformed this data by dividing the 4097 data points into 23 chunks per individual, each chunk representing 1 second of brain activity with 178 data points. This restructuring resulted in 11,500 rows, each containing 178 data points and a label (y) in the 179th column. The labels are as follows:
- 1: Seizure activity
- 2: Tumor area
- 3: Healthy area
- 4: Eyes closed
- 5: Eyes open

Our objective is to classify these EEG recordings to aid in the detection and prediction of seizure activities.

# Methodology
## Data Preprocessing

1. **Data Cleaning:** We began by removing any missing value and unnecessary columns in our dataset. This step ensured the integrity and consistency of our dataset.

2. **Normalization:** Each EEG data point was normalized to ensure that the values across different recordings were comparable. We used Z-score normalization to standardize the data, which involved subtracting the mean and dividing by the standard deviation for each data point.

3. **Target Variable:** We change our approche from multiclass classification to binary classification to predict `Seizure` and `not Seizure` activity  

> **Future apprache:**   
    1.**Handling Imbalanced Data:** Given the varying distribution of labels (e.g., more 'not Seizure' cases than 'seizure' cases), we applied techniques like oversampling and Resampling the minority classes to balance the dataset.

## Feature Extraction

1. **Statistical Features:** From each 1-second chunk of data, we extracted statistical features like mean, median, standard deviation, and skewness. These features provide a basic understanding of the distribution and variability of the EEG signals.

2. **Correlation:** We Extract Pearson correlation Coeffitient for each pair of features. These features provide a basic understanding of the relation of features based of variation.
> **Future apprache**:   
    1.**PCA:** Given the high dimensional dataset that we have and multicollinearity that we have we can use PCA to decrease dimension of our dataset to make the comlexity of our dataset much less.  
    2.**Clustering:** We can use advantagous of Clustering algorithms like: Kmeans, Meanshift, DBScan... to identify potential clusters in our dataset and also potential noises

The combination of these preprocessing and feature extraction techniques was critical in preparing the dataset for subsequent machine learning models. They allowed us to capture the essential characteristics of the EEG signals relevant for distinguishing between different states, including seizure activity.

## Machine Learning Techniques and Algorithms

In this project, we employed a variety of machine learning techniques to classify EEG signals into different categories, including seizure activity and not seizure activity. The following is an overview of the key algorithms and approaches we used:

### 1. Supervised Learning Algorithms:

- **Support Vector Machine (SVM):** We used SVM for its effectiveness in handling high-dimensional data. It was particularly useful in classifying the EEG signals into the respective categories by finding the optimal hyperplane that separates the different classes.

- **Random Forest Classifier:** This ensemble learning method was utilized for its robustness and ability to handle non-linear data.

- **KNN:** K-Nearest Neighbors (KNN) is a straightforward and effective machine learning algorithm we used classification. It identifies the k closest data points in the training set to a given data point and makes predictions based on these neighboring points. KNN is particularly notable for its ease of implementation, interpretability, and strong performance on a variety of problems.

- **Gradient Boosting:** Gradient Boosting is a sophisticated machine learning technique used for classification. It builds a model incrementally from an ensemble of weak learners, typically decision trees. By focusing on correcting the mistakes of previous learners in the sequence, it requires careful tuning of parameters and can be computationally intensive. It's especially useful in scenarios where prediction accuracy is paramount.

- **Logistic Regression:** By Logistic Regression we can predict the probability of an event's occurrence by fitting data to a logistic function. Think of it as a way to draw a line (or a curve, in more complex scenarios) that separates two classes of data points. It's particularly useful for understanding the impact of several independent variables on a binary outcome, like 'yes' or 'no' or `seizure` activity and `not seizure` activity.

We also used two method of ensemble models as `Voting` and `Stacking`. 

### 2. Unsupervised Learning for Feature Learning: `(Future Steps)`

- We can rich our project by using **Autoencoders** and Machine Learning clustering method like **DBScan** and **Kmeans** in Future steps  

### 3. Model Evaluation and Selection:

- **Performance Metrics:** Accuracy, Precision, Recall, and F1-Score were the primary metrics used to evaluate the models. Given the critical nature of seizure detection, we focused extensively on maximizing recall and F1-Score to reduce false negatives. We utilized Gridsearch score based of k-folds of dataset

### 4. Hyperparameter Tuning:

- **Grid Search:** These method was used for hyperparameter optimization. By systematically working through multiple combinations of parameter tunes, it helped in finding the most effective model settings. Also we use its validation score of kfolds dataset as a metric of our model performance 

This comprehensive approach to machine learning ensures robustness and accuracy in our EEG signal classification. Our methodology emphasizes not only predictive performance but also the interpretability of results, which is vital in medical applications like seizure detection.

# Results

My project's exploration into EEG data analysis using various machine learning models has led to some noteworthy insights and conclusions. Here are the summarized results and our interpretations:

|         Model         |  Accuracy  |  Recall  |    AUC    | Precision |  F-Score  |
|-----------------------|------------|----------|-----------|-----------|-----------|
|       Stacking        |   0.980290 | 0.943478 | 0.966486  | 0.957353  | 0.944004  |
|         SVC           |   0.979130 | 0.936232 | 0.963043  | 0.958457  | 0.937068  |
|        Voting         |   0.961159 | 0.831884 | 0.912681  | 0.969595  | 0.836453  |
|     RandomForest      |   0.959420 | 0.830435 | 0.911051  | 0.961409  | 0.834809  |
|         KNN           |   0.926087 | 0.630435 | 0.815217  | 1.0       | 0.639525  |
| Logistic Regression   |   0.822319 | 0.111594 | 0.555797  | 1.0       | 0.115542  |

### Key Insights:

1. **Superiority of Ensemble Methods:** The Stacking model, combining multiple algorithms, outperformed individual models. This highlights the advantage of ensemble methods in capturing diverse patterns in the data, leading to more robust predictions.

2. **Handling Imbalanced Data:** The notable performance of our models, especially the Stacking and SVC, on an imbalanced dataset emphasizes the importance of appropriate feature engineering and model selection over the mere balancing of data. This suggests that in certain scenarios, the traditional focus on dataset balancing can be reconsidered.

3. **Model Complexity vs. Performance:** The performance of the Stacking model suggests that while complex models can offer better results, they require careful tuning. Simpler models like SVC and RandomForest, with the right adjustments, can still provide near-comparable performance, highlighting a trade-off between model complexity and ease of implementation.

4. **Precision-Recall Dynamics:** The contrast between high precision and low recall in models like KNN and Logistic Regression underscores the challenge of the precision-recall trade-off in machine learning. In medical applications such as seizure detection, a balance between these metrics is crucial, as both false negatives and false positives carry significant consequences.

5. **Reliability of Recall as a Metric:** Given the critical nature of seizure detection, the high recall rates of the top-performing models (Stacking and SVC) are encouraging. It indicates these models' effectiveness in identifying true seizure events, which is paramount in a medical context.

6. **Potential for Real-World Application:** The success of these models, particularly in terms of AUC and Recall, positions them as promising tools for real-world applications in the medical field. They hold the potential to aid in early and accurate seizure detection, which is vital for patient care and intervention.

In conclusion, our analysis not only demonstrates the effectiveness of machine learning in EEG signal classification but also provides valuable insights into the nuances of model selection and performance evaluation in the context of medical data analysis. The project's findings have significant implications for advancing seizure detection methods, contributing to the broader goal of enhancing patient outcomes in neurology.

# Potential Business and Practical Applications

The outcomes of our EEG-Based Seizure Detection and Analysis project have significant implications not only in the medical field but also in various business and practical contexts. Here are some key areas where our findings can be applied:

1. **Healthcare Industry:** The primary application of our project is in the healthcare sector. Hospitals and clinics can use these models to enhance their diagnostic tools, providing more accurate and timely detection of seizure activities. This can lead to better patient care, reduced hospital stays, and potentially life-saving early interventions.

2. **Wearable Health Monitoring Devices:** There's a growing market for wearable health technology. Our models can be integrated into wearable EEG monitoring devices, providing continuous, real-time analysis of brain activity. This can be particularly useful for patients with epilepsy, allowing for immediate response in case of seizure detection.

3. **Telemedicine and Remote Monitoring:** Our project's findings can be leveraged in telemedicine platforms, offering remote monitoring services for patients with neurological disorders. This not only makes healthcare more accessible but also provides a continuous stream of data for better management of the patient's condition.

4. **Pharmaceutical Research and Development:** In pharmaceuticals, our models can assist in research and development, particularly in testing the efficacy of drugs meant to treat seizures or other neurological conditions. By analyzing EEG data, researchers can gain insights into how different treatments affect brain activity.

5. **Data-Driven Patient Care:** Our models can also aid in data-driven patient care strategies. By analyzing EEG data, healthcare providers can develop personalized treatment plans, monitor patient progress more accurately, and make informed decisions about patient care.

6. **Education and Training:** The methodology and findings of our project can be used as educational tools in academic settings, especially in courses related to data science, machine learning, and neurology. This can help in training the next generation of data scientists and healthcare professionals.

In conclusion, the successful application of machine learning techniques in EEG signal classification has broad implications across various industries. By improving seizure detection and offering deeper insights into neurological conditions, our project stands to make significant contributions in both the business world and in enhancing the quality of life for individuals with neurological conditions.

# Next Steps

The next phase of the EEG-Based Seizure Detection and Analysis project will focus on expanding the scope and depth of our current findings, with the aim of enhancing the predictive accuracy, generalizability, and real-world applicability of our models. The following steps are outlined to guide future work:

1. **Deep Learning Integration:** Incorporate deep learning models such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to capture more complex patterns in the EEG data that traditional machine learning methods may miss. This includes exploring architectures specifically designed for time-series analysis and sequence modeling.

2. **Expand Dataset and Diversify Sources:** To improve the robustness and generalizability of our models, we plan to integrate additional datasets from diverse demographics and conditions. This will also involve seeking collaborations with medical institutions for access to a broader range of EEG recordings.

3. **Real-Time Analysis Implementation:** Develop a framework for real-time seizure detection, which is critical for wearable devices and patient monitoring systems. This includes optimizing models for low-latency predictions and ensuring complex models can run on devices with limited computational resources.

4. **Clinical Trial and Validation:** Partner with healthcare providers to conduct clinical trials, assessing the practical effectiveness of our seizure detection system in a real-world setting. This step is crucial for understanding the models' performance outside controlled environments and gathering feedback for further improvements.

5. **Enhance Data Preprocessing and Feature Engineering:** Explore advanced signal processing techniques and feature extraction methods to better capture the nuances of EEG data. This may include wavelet transforms, higher-order statistics, and exploring new correlation metrics between EEG channels.

6. **Continuous Learning and Model Updating:** Implement mechanisms for continuous learning, where the models are regularly updated with new data, ensuring they remain accurate and relevant over time. This includes developing a pipeline for automatic retraining and validation of models.

By following these steps, we aim to not only enhance the technical capabilities of our seizure detection system but also to ensure it can be effectively integrated into healthcare practices, ultimately contributing to improved patient care and outcomes in epilepsy management.


# Contributing

## Contributing to EEG-Based Seizure Detection and Analysis

I highly appreciate contributions and are excited to collaborate with the community on this data science project. Whether it's through data analysis, model improvement, documentation, or reporting issues, your input is valuable. Here’s how you can contribute:

1. **Fork the Repository:** Start by forking the project repository to your GitHub account. This creates a personal copy for you to work on.

2. **Clone the Forked Repository:** Clone the repository to your local machine. This allows you to make changes and test them locally.

   ```git
   git clone https://github.com/mrpintime/Epileptic-Seizure-Recognition.git
   ```

3. **Create a New Branch:** Create a new branch for your work. This keeps your changes organized and separate from the main branch.

   ```git
   git checkout -b feature-or-fix-branch-name
   ```

4. **Contribute Your Changes:**
   - **Data Analysis:** If you’re adding new analysis, ensure your code is well-documented and follows the project’s coding conventions. Include comments and README updates explaining your methodology.
   - **Model Improvement:** For changes to existing models, provide a clear explanation and any performance metrics or results to support the improvements.
   - **Data Contribution:** If contributing new data, ensure it is properly cleaned, formatted, and accompanied by a source description.

5. **Commit and Push Your Changes:** Commit your changes with a clear message describing the update. Push the changes to your forked repository.

   ```git
   git commit -m "Detailed description of changes"
   git push origin feature-or-fix-branch-name
   ```

6. **Create a Pull Request:** Go to your fork on GitHub and initiate a pull request. Fill out the PR template with all necessary details.

7. **Code Review and Discussion:** Wait for the project maintainer(that's me 😁 ) to review your PR. Be open to discussion and make any required updates.

### Reporting Issues

- Use the Issues tab to report problems or suggest enhancements.
- Be as specific as possible in your report. Include steps to reproduce the issue, along with any relevant data, code snippets, or error messages.

### General Guidelines

- Adhere to the project's coding and data handling standards.
- Update documentation and test cases for substantial changes.
- Keep your submissions focused and relevant to the project's goals.

Your contributions play a vital role in the success and improvement of EEG-Based Seizure Detection and Analysis. We look forward to your innovative ideas and collaborative efforts!

# License
- This project is licensed under the MIT License - see the `LICENSE.md` file for details.

# Contact
- Contact me through my Linkedin: https://www.linkedin.com/in/moein-zeidanlou

# Acknowledgments
- Credits to UCI Machine Learning Repository for providing the dataset.  
Dataset Link: https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition
