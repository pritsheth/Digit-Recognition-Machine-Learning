
***
Welcome to the Digit-Recognition-Machine-Learning wiki!

# KNN Classifier for digit recognition:

In k-NN classification, we will observe the class label of nearest neighbors and will take the majority of votes. Based on the K value we can decide how far we have to go for exploring nearest class labels. 

# Task of building the classifier for Digit Recognition:

Each example includes 28x28 grey-scale pixel values as features and a categorical class label out of 0-9. You can manually download the dataset from Dr. Yann Lecun’s webpage (http://yann.lecun.com/exdb/mnist/)
or automatically get it from some libraries/packages .
(e.g., as done in section 5.9 of http://scikitlearn.org/stable/datasets/index.html for sklearn in Python).
I have not used any existing class or functions (e.g., sklearn.neighbors.KNeighborsClassifier) and implemented it from scratch.

				Digit_label, 28*28 pixel values [0-255]

Below is the size of given Input dataset : 

Train: 60000 images 
Test: 10000 images

Image Type : Pixels and gray scale.

Algorithm :

- Covert image data into csv files.
- Load training data and test data into list.
- For each test sample in test set:
    
      - Start :

            - compute the distance metrics : [subtract the test instance with train_instance and square it, do the 
            row-wise sum to get the actual distance]
            - we will have all the distance from one test instance to all training instances
            - Sort the distance vector
            - Now for each K value, 
                Start: 

                        - do the slice of K length from distance vector and compute the prediction.
                        - Based on the majority vote decide the target class
                        - If got the incorrect label then increment the error score for that particular K value. 
                - End
          - End
- Use the error vectore and divide it by length of test data.
- Plot the graph.




Given below, results I obtained with different K values for KNN :

k = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
ErrorRate = [0.096, 0.091, 0.108, 0.123, 0.134, 0.136, 0.143, 0.149, 0.158, 0.166, 0.171]

![KNN](KmeansClustering/KNN_graph.png?raw=true "K values vs error rate")


	            K = 1, with Euclidean : 
	            Error rate : 9.6 %

	            K = 9, with Euclidean :
	            Error rate : 9.1%

	            K = 19,  with Euclidean :
	            Error Rate : 10.8 % 

	            K = 29 , with Euclidean :
	            Error Rate : 12.3 %

	            K = 39, with Euclidean :
	            Error Rate : 13.4%

	            K= 49, with Euclidean :
	            Error Rate : 13.6 %

	            K = 59, with Euclidean :
	            Error Rate :  14.3 %

	            K = 69, with Euclidean :
	            Error Rate : 14.9 %
	            
	            K = 79, with Euclidean :
	            Error Rate : 15.8 %
	            
	            K = 89, with Euclidean :
	            Error Rate : 16.6 %
	            
	            K = 99, with Euclidean :
	            Error Rate : 17.1 %



# Problems faced when optimizing KNN code:

The real challenge was to compute the distance matrix for each test instance and the value of the K. It will definitely take more time if we don't use the vectorized form of the training and testing data. Normal brute-force
will take almost 20 minutes to run this algorithm.

# Optimizations, I did to solve above problem and better performance: 

1) Reducing pixels [converting into feature vector]: MNIST data was having 60,000 rows of digits in training set which was having 28*28 pixel. I have converted that into 784 features so that it would be easy to process in our algorithm otherwise it would take much more time in training phase.  

2) Following optimization to reduce the time of training phase. 

     ### To substract it from all the target points in constant time
        `sub = test_arr[1:] - train_arr[:, 1:]`
       
      ### To make it euclidean distance :(we dont have to compute the square root for better perfomance)
        `sqr = np.square(sub)`
        `dist = np.sum(sqr, axis=1)`
      
# Conclusion :

KNN is lazy classifier and it helps when you dont want your model to be re-trained from scratch every time you add the new training samples to the set on the fly. It is time complexity is T*N*LOGN (where T= total test samples, N = total training samples, as we compute the distance for each test point to all training samples). We observed that as we increase the value of K, our error rate will increase too.

# Parameters and Results :

Image encoding: Grayscale
Distance measure: Euclidean
Accuracy:  between 80 % and 91%
Execution time: 51.46 seconds

You can run the program as follows:

- Clone the repository
- Open Kmeans Classifier module
- Run KNN classifier ($ python KNNClassifier.py )
- Make sure you have the mnist_test and mnist_train csv files in the same folder. [Otherwise, run the convert method to create the csv files]
- Check the time, plotted graph, error_rate metrics
***
