# SpotifyRecommendations

## Introduction
This project involves building a recommendation system for a Spotify-like application, using data from approximately 2000 users and nearly 20,000 artists. The recommendation system will predict the number of times a user will listen to a specific artist based on past listening data and optionally using a social network that describes relationships between users.

## Data Description
- **artist_user.csv**: Contains partial data of artist listenings by users. Each row has a user identifier, artist identifier, and a "weight" which represents the number of times that user has listened to that artist.
- **test.csv**: Contains pairs of users and artists for which the number of listenings needs to be predicted.
- **friends_user.csv**: Describes the social network between users. Each row contains a pair of users who are friends. This file is optional for use in the recommendation system.

## Objectives
There are two main tasks in this project:
1. **Task 1**: Minimize the total penalty without using the social network. Only past listening data should be used.
2. **Task 2** (Optional): Minimize the total penalty using both past listening data and the social network. This task is more challenging and thus, carries a bonus for successful implementation.

## Prediction Evaluation
The predictions will be evaluated based on the logarithmic difference between the predicted and actual number of listenings, using the formula:
\[ l_{ui} = (\log_{10} \hat{r}_{ui} - \log_{10} r_{ui})^2 \]
The total penalty is the sum of all individual penalties:
\[ L = \sum l_{ui} \]
where \( \hat{r}_{ui} \) is the predicted value and \( r_{ui} \) is the true value (which is unknown to you).
