# Final Analysis Conclusions
## Did you find the results corresponded with your initial belief in the data? If yes/no, why do you think this was the case?

In our analysis, we tested quite a few hypotheses, each of which interrogated one of our initial assumptions about the general sentiment around work-from-home policies. 
A large part of our hypothesis testing centered around looking for differences between our article sources. We had assumed that the sentiments within a category of sources would be similar and that the sentiments between different categories of source would be different, with software developers being more positive regarding the idea than the media. Of these assumptions, only one appeared to be true. When we compared the average sentiments between different subreddits and between different news sources, we failed to reject the null hypothesis that the average sentiment was the same in both cases, suggesting that our assumption that each category would be aligned in their findings was correct. However, we also failed to reject the null hypothesis that reddit and the news sources would have the same average sentiment, suggesting that we were incorrect to assume that software developers would be more positive when talking about work from home than the news would be. 
While there are many possible explanations for these results, the one we think is the most likely is that there is a single source of bias underpinning all of the sources we examined: a negativity bias. Of the sentiment ratings we generated, only 29% were positive and 14% were neutral, leaving 57% as negative. Upon further reflection, this makes sense; the subreddits we sampled were largely used for advice, and people generally do not need advice when things are going well. Additionally, negative events are generally more likely to be newsworthy, especially as much of the news surrounding WFH is in the context of the COVID-19 pandemic which is an inherently negative subject. Thus, it is likely that this negativity bias had a larger effect on the data than any causes of variance between our sources. This conjecture also aligns with the output of our regression model, as it found a slight negative slope with a high prediction accuracy and low r2 value, suggesting that the date of publication was not a significant factor in the sentiment of the articles, a result we found somewhat surprising but that would align with this conjecture.

## Do you believe the tools for analysis that you chose were appropriate? If yes/no, why or what method could have been used?
We believe that the tools we used for our analysis were generally appropriate, though it is also clear that there is substantial room for improvement. For our ML methods, our sentiment analysis model generally had high precision and we qualitatively agreed with its output the vast majority of the time; however, it had a significant bias towards rating news articles neutrally. Future work could iterate on this result by integrating different models that are better-trained on news articles or by hand-labeling enough data to train a model from scratch, two approaches that we did not have time for here. Our linear regression model had a high prediction accuracy but a low r2 value, suggesting that it was appropriate for predicting the general trend in sentiment but failed to capture more nuanced variance in the data. Future work could experiment more with using more complex forms of regression, such as logistic regression or a deep learning regression model. Further, our dataset was limited by the fact that it started in 2020 during the COVID-19 pandemic when remote work surged in popularity, so including data from a few years prior to 2020 might help us to capture trends better. However, given our dataset it made sense to stick with a simple model as linear regression provided us with, so we conclude that the tools we used for analysis were appropriate but that there is also always room for improvement. 
For our hypothesis tests, we heavily utilized the two-sample T-test. We believe that this was the most appropriate tool we had available due to our hypotheses consisting entirely of comparing the same statistic between two different samples. Although there are other methods that could have been used to test the second, third and fourth hypotheses, the two-sample T-test was the most appropriate as our data is normally disturbed data. Non-parametric tests like the Mann-Whitney U-test, and Kruskal-Wallis test are not the best options for normally distributed data and also would have been more inconvenient to implement due to the lack of a python package with them in it. Hence, we are pleased with the results from the two-sample T-tests and do not believe that there is a more appropriate testing method for the second to fourth hypotheses.

## Was the data adequate for your analysis? If not, what aspects of the data were problematic and how could you have remedied that? 
For the most part, our data was adequate for our analysis. We had enough samples to test the hypotheses that we wanted to investigate, our data was not skewed towards extreme values, and the values that we did have were qualitatively believable and quantitatively precise. However, there was a neutrality bias in the sentiment analysis ML algorithm that we described above that made its outputs on the news articles somewhat less reliable. Addressing this would require performing a lot of additional iteration on which model(s) to use, which was not something we ultimately had time for.
Moreover, the negativity bias in the sentiment ratings, in which a large portion of the sentiment ratings were negative, overpowered any variance between the reddit and news articles. However, there is no need to remedy this phenomenon as it represents the reality of the situation. Negative sentiments toward remote work are rampant and make the sentiments across different mediums uniform. Hence, small subtleties and differences in sentiments between reddit and newspapers would always be overshadowed by this trend.
The p value being relatively close to 0.05 for the hypothesis checking if the subreddits are uniform in terms of sentiment ratings is quite interesting. It may be helpful to collect more data from these subreddits to check whether the p value converges to 0.05 or less when the quantity of data increases. If the p value reaches 0.05 or less, that would imply that the subreddits are not entirely uniform in terms of their sentiment ratings, which is contrary to our prediction. However, this exploration is not necessary as our prediction was already confirmed by our p value being greater than 0.05.
Another metric that could be interesting to examine is the sentiment difference between companies themselves and their employees; one could argue that employees would be generally more in favor of remote work, whereas organizations might prefer in-person work to try and squeeze maximum productivity out of their employees. It would also be interesting to analyze how remote work sentiment differs across rural and urban areas, and high-income and low-income families. 