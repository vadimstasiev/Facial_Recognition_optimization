# Hyper Parameter Tunning

To run: `python run.py`

To start the tensorboard: `tensorboard --logdir .\logs\hparam_tuning\`

## Introduction

The success of a machine learning solution often depends heavily on the choice of good hyper-parameters.
### What’s covered?
This report addresses the objectives of the second assignment which serve as an improvement to the solution developed in the first, more precisely, the brief outlines the need to use one or more strategies to explore the different combinations of hyper parameters, ideally attain one that yields better results. The results that are being sought are ones that surpass the previously developed recognition solution in at least efficiency if not accuracy, given that the accuracy was already high.


## Designing a Solution

As mentioned above, for this assignment it is required the use of at least one of the following suggested strategies:

	Random Search

 -	Random search is a technique through which random hyper parameter values are tested using a uniform random parameter generator, which ensures that random values are generated uniformly which ensures that different values are tested. (Bengio & Bergstra, 2011)
    
	Meta Learning

 -	In this context, this is referring to the use of an existing model that was trained to optimize the hyper parameters of many different models to optimize the current model. (Hospedales, et al., 2020)
    
	Adaptive Boosting

 -	Also known as “AdaBoost”, it is a machine learning technique that aims to fit models with repeatedly modified versions of the data where the data modifications are made to increase the chances of the model correctly classifying previously incorrect classifications. (sci-kit learn, 2020)
    
	Cascade Correlation

 -	This is a supervised learning algorithm that instead of using a fixed topology, it starts off with a minimal network and automatically trains and adds new hidden layers one by one. This allows the network to learn quickly as well as determine its own size and topology. (Fahlman & Lebiere, 1993)



The original solution used an exhaustive search, also known as “grid search”, this was previously used to find the configuration with the best performing number of epochs and neurons, however, for this assignment, it was decided to also add a dropout layer and use its value as another hyper parameter as well as adding the ‘sgd’ and ‘nadam’ as alternative values for the optimizer, which would also be tested individually during the hyper parameter evaluation.

To start with, it was decided to rewrite the grid search to also include the aforementioned added parameters, to recap on what that is and does, grid search is the process through which it is attempted every single combination of the different hyper parameters. It works iteratively through all of the different combinations, making this the ideal, yet unpractical, solution for finding the absolute best hyper parameter combination.

This time around the implementation also revolved on using a tool provided by “TensorFlow” named “TensorBoard”. (TensorFlow, 2021) This tool provides a dashboard for viewing all the results of all of the different runs for easy and fast comparison, however the code had to also be adapted to generate compatible data that can actually be interpreted by this tool. 

Bellow it is shown the reimplementation of the grid search to generate the mentioned compatible data for interpretation by “TensorBoard”.

![image](https://user-images.githubusercontent.com/17814261/190217906-70eb351a-c15e-40c4-b0f6-749047895b29.png)


![image](https://user-images.githubusercontent.com/17814261/190217745-6cd18390-1f53-4168-bb2c-56918094a864.png)


Above it can be seen that all the parameters are being generated and ran iteratively, the function “run” creates the model with the respective parameters and records all the relevant information to the disk.
As already mentioned, the grid search technique is unpractical and this time around due to the addition of the previously mentioned hyper parameters the run took too long so it was eventually cancelled, this also proved the point as to why this technique has hardly any real-world application since any modern machine learning solution, that serves any kind of purpose, would have at least this level of complexity as far as hyper parameters go.
Moving on to using at least one of the suggested strategies, for this assignment, it was decided to use “Random Search” since the implementation of it seemed fairly straight-forward and promising, as long as the parameter values would be uniformly generated such as to ensure no single value was tested more than the others. (scikit learn, 2021) In order to do this, the following function was created:

![image](https://user-images.githubusercontent.com/17814261/190217701-7d90d8e2-b3cf-46c4-865a-c5ec2ecaac3f.png)


The function was then implemented in the following way:

![image](https://user-images.githubusercontent.com/17814261/190217639-eca6f7f9-f91d-48e1-9f00-402e53e97e1a.png)


As it can be seen above, there is a for loop that runs 10,000 times and over each run, different random hyper-parameters are generated uniformly, the run results are then also saved for later analysis.


## Unexpected Issues
Despite looking like a promising tool, TensorBoard appears to have a bug that causes a memory leak which according to the community has been present in the software for at least a few years (TensorFlow GitHub, 2021), and has not been fixed yet, this seems to be triggered whenever TensorBoard has to load a lot of logs, the number of logs tested in here were roughly 10,000 and it seemed to cause the program to fill up the entire system memory (RAM) in the space of a few seconds, naturally if not stopped, the memory leak would then overflow onto to windows paging file. This issue might be related to the windows NTFS file system as it is a known file system to have many issues when dealing with multiple directories nested within a single one but that was not pinpointed, it was instead found a work-around, to work around the issue the “TensorBoard” software needs to be started on an empty directory and then after the fact, move the logs into the working directory.

![image](https://user-images.githubusercontent.com/17814261/190217532-b0bb2e18-8d62-452a-9a2c-162ae860a8ed.png)


## Experiments
As previously mentioned, the experiments conducted using grid search were cancelled so they will not be included.
As for the random search experiment, this was concluded with 10,000 runs with the following tested values per parameter:
![image](https://user-images.githubusercontent.com/17814261/190217429-aeb56b17-3067-44fb-9352-6ee7bf38c069.png)


As these same values were used in the original grid search, math would suggest that it would have resulted in a staggering 238,260‬ runs, so while 10,000 runs may seem much, it actually equates to roughly only 4% of the total number of combination possibilities.
The resulting graph of all the runs can be seen bellow:

![image](https://user-images.githubusercontent.com/17814261/190217327-f377e5ef-43d8-4bf6-a78a-f944b9a415b7.png)

![image](https://user-images.githubusercontent.com/17814261/190217242-9e6a5ff0-ac86-46e5-9d06-a8fa34bf8672.png)


While the graphs above can give a rough idea of how the different parameters affected the model’s accuracy, it looks fairly dense and therefore it is hard to drawn conclusions from it, to do that we must instead look at the top results and compare them to each other.
On the TensorBoard, with all the runs selected, under the HPARAMS tab, there is some functionality that allows the data to be sorted, in order to see what the best results were, this data needs to be sorted by the accuracy in a descending direction.
![image](https://user-images.githubusercontent.com/17814261/190217206-34a76c2e-1a0b-480a-86de-080e996ae71c.png)


On the resulting table it can be seen the best results of all of the 10,000 experimental runs and while it may not always be the case, in this case, the top result seems to not only have the same accuracy as the few before it, but it also does so by being the most efficient one as it has the least number of hidden neurons, hitting an accuracy of 98% with only 154 neurons.
Another experiment was conducted using more optimizers but only with 1,000 runs, and while due to this small amount of runs, generation a small amount of different combinations, the findings of these results cannot be taken with certainty.
More specifically, the parameters tested were:
Epochs	From 12 to 40
Neurons	From 50 to 240
Dropout Rate	0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.17, 0.18, 0.19, 0.2
Optimizer	'adam', 'nadam', “RMSprop”, “Adamax”
The resulting accuracy graph was as follows:
![image](https://user-images.githubusercontent.com/17814261/190217065-1b4d2c2d-02b8-4b05-8b5f-9c8aa29b7806.png)

This time, the accuracy was filtered to only show results with accuracies above 95% and as it can be seen from the small sample, nadam continues to reach the highest accuracy from the run samples, however it seems that the newly tested “RMSprop” optimizer produced a run with an accuracy higher than any of the “adam” combinations, which in the previous testing seemed to be highly competitive with “nadam”, this might mean that there might be a combination where the “RMSprop” optimizer might actually surpass the current highest accuracy which uses “nadam”.

## Conclusion

After many failed efforts and strange computing occurrences the requested code was finally produced and likewise the experiments ran, therefore fulfilling the requirements of the task.
With the ending of the experiments it concludes the assignment task, it also concludes the fact that using strategies that don’t require brute force to find answers are far more effective, not to mention, recommended, similarly, other untried strategies could mean leaps in the progress of improving the solution.

To improve on this work there are further steps to take, that include: meta-learning, adaptive-boosting and other machine learning techniques, not to mention further tweaking to the model in an effort to finding the ideal architecture that produces an accuracy as close to the truth as possible, or perhaps even more plausibly, a high accuracy but efficient model that is targeted to run on portable low-consumption hardware.


