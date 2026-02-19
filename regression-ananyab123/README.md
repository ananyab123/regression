[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/u1oODY1i)
# README

Tell us about your implementation!
I completed the linear regression class, where I added the closed-form least square equation for fitting, prediction, and MSE calculation (determine model accuracy). I implemented polynomial regression and ridge regression as well, which actually adjusted the linear model with regularization to avoid overfitting. I also implemented in the bias_and_variance class to finish th methods for sampling, building estimators, calculating bias, calculating variance, and calculating MSE.

Tell us about your tests! None because no test file!

Answer the conceptual questions!

Task 1.3
When I ran the correlation analysis, some features that were related to life expectancy were income composition of resources, schooling, and adult mortality. The most positively correlated pair overall was income composition of resources and schooling at around 0.92, and both are strongly correlated to life expectancy, which is reasonable because better income distribution and longer schooling in countries generally also have longer lifespans. Meanwhile, adult mortality had one of the strongest negative correlations with income composition of resources at around -0.67 and schooling at around -0.57, which adheres to what I expected because higher mortality is linked to lower life expectancy. It was also interesting to me that BMI and the thinness features were negatively correlated at around -0.55 (makes sense because populations with lower BMI usually have higher thinness rates).

Task 1.4
Feature: Schooling
Train MSE: 26.1560
Test  MSE: 31.0694
Feature: Income_composition_of_resources
Train MSE: 14.7427
Test  MSE: 17.9567
Feature: Adult_Mortality
Train MSE: 29.4493
Test  MSE: 29.1344
I chose to focus on schooling, income composition of resources, and adult mortality. The first two show strong positive relationships with life expectancy, while adult mortality has a negative relationship with life expectancy. To visually summarize my models, income composition of resources had the cleanest linear pattern and it also had the lowest training and test errors (as seen above), indicating the best fit. Then, schooling also had a relatively good positive trend but with higher errors as seen, while adult mortality had a negative slope, indicating that higher mortality lowers life expectancy.

Task 3.2
To compare the bias and variance of various 7th degree polynomial regression models, I use the following values for λ: 0.01, 1, and 10. The smaller λ was, the more models looked almost identical to standard polynomial regression (wiggly and overfit to the noise). However, I noticd that as λ increased, they got smoother and more stbale because we reduced the variance between estimators. However, as λ increased, the average estimator flattened, showing higher bias. Thus, a smaller λ has lower bias and higher variance, but a greater λ had higher bias and lower variance.

Task 4.1
In the standard regression, some coefficients were very large and unstable, such as alcohol (165.22), polio (-165.21), and GDP (-63.53). Thus, there was overfitting. Meanwhile,ridge regression with λ = 0.1 reduced those with alcohol at -2.26, polio at -4.67, and GDP at -24.60, and all of those features also stayed in the model. However, with LASSO, most of the coefficients were brought to 0 and only the strongest predictors stayed, such as adult mortality (–13.55), income composition of resources (32.26), and total expenditure (3.57). Thus, compared to ridge regression, LASSO actually removes weak features and regularizes more strongly, but both are still different from standard regression. 

Hours taken: 15 hrs

Collaborators: none

Known Bugs: none

AI Use Description: none

You must acknowledge use here and submit transcript if AI was used for portions of the assignment