# Medior Machine Learning Engineer

## Description 

In order to run a successful promotion campaign at Albert Heijn, it is important to obtain an understanding of the uplift in sales of certain items when put on promotion. This is not only necessary to decide on what type of promotions to run, but also to get an idea on the required levels of stock to discuss with our suppliers. 

The supplied notebook builds a model that serves as input to a planning tool in which they want to see the effect of promotions for the upcoming weeks. For simplicity, any EDA and hyperparameter search (using a validation set) have been omitted, and we are using a simplified dataset. For this MLE case, you do not need to worry about model performance. 

Your job as machine learning engineer is to transform this solution into a proper application that can be used in production. Although there are many things you can do, we advise to not spend more than 8 hours for the solution. We also value your free time! 

You’re not expected to deliver the perfect application in this short period. We do expect you to list your assumptions and provide comments on your work. It’s important to show the aspects that you think are most important. 



## Requirements 

- We expect that your solution works out of the box. 
- Provide a README file that describes how the application works, and any parameters required for the solution to function. 

- Make sure to split the solution into training and inference. 

- Train and evaluate the model. Do not spend time on hyperparameter tuning, instead use the notebook for this. 
- The output of the model should be in actual units (see last cells of notebook). 
- The solution should be able to accept new input values (see last cells of notebook). 
- The solution can be a one off-run or served as an API. 

- An architecture diagram, describing how your solution could be deployed. 
- During the technical interview, be prepared to discuss how the CI/CD pipelines for your solution could look like. 



## Pointers 

The pointers below may help you build your application according to our expectations. Feel free to disregard these and do things your own way if your preferences differ or if you prefer to focus on improving other parts of your application. 

- We would like to see district components for the training and inference. 

- A good way to structure your Python source code is to set up a [Python package](https://packaging.python.org/en/latest/tutorials/packaging-projects/). 
- When working together on an application, adhering to a consistent coding style is important. Consider including a way to enforce style standards for your code. 
- Consider creating a Dockerfile – that will help you ensure the solution works in our MacBooks as well. 



## Deliverables 

You are expected to deliver your solution packaged into a single ZIP file. The ZIP file should contain everything required for us to be able to run and review your solution. 



## Evaluation criteria 

Your solution will be evaluated based on the following criteria: 

- Structure of the project. 
- Code quality (I.e., adheres to software engineering principles). 
- Usability (I.e., how easy is to use the application after following the instructions). 

- Separation of concerns. 
- Positive surprises are welcomed and can earn bonus points. J 