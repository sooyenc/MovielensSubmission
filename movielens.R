# title: "HarvardX: PH125.9x Data Science: Movielens Project Submission"
# author: "Chang Soo Yen"
# date: "6 January 2020"
# output: pdf_document

## 1 Introduction ##
  
## 1.1 Dataset ##

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
```

## 1.2 Project Goals ##

## 1.3 Key Steps ##

################################
# RMSE Function
################################

RMSE <- function(actual_rating, predicted_rating){
  round(sqrt(mean((actual_rating - predicted_rating)^2)),10)
}

## 2 Methodology ##

################################
# View head of edx dataset
################################

head(edx)

################################
# View structure of edx dataset
################################

str(edx)

################################
# Check for any NA values in edx dataset
################################

any(is.na(edx))

################################
# View the number of unique userIds and movieIds
################################

edx %>% 
  summarize(distinct_users = n_distinct(userId),
            distinct_movies = n_distinct(movieId))

## 2.1 Introduction to EDX Dataset's Subsets ##

################################
# Create tempedx set, tempvalidation set from edx dataset
################################

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
tempedx <- edx[-test_index,]
temporary <- edx[test_index,]

# Make sure userId and movieId in tempvalidation set are also in tempedx set

tempvalidation <- temporary %>% 
  semi_join(tempedx, by = "movieId") %>%
  semi_join(tempedx, by = "userId")

# Add rows removed from tempvalidation set back into tempedx set

removed <- anti_join(temporary, tempvalidation)
tempedx <- rbind(tempedx, removed)

## 2.2 Average Movie Rating ##
  
################################
# View distribution of rating in edx dataset
################################

edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "red") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating Distribution")

## 2.2.1 Model ##

################################
# Find mean rating of tempedx dataset
################################

mu <- mean(tempedx$rating)
mu

################################
# Calculate rmse from average moving rating model
################################

movierating_rmse <- RMSE(tempvalidation$rating, mu)
movierating_rmse

################################
# Create tibble to store RMSE
################################

options(pillar.sigfig = 5)
rmse_results <- tibble(method = "Temp Average Movie Rating", RMSE = movierating_rmse)
rmse_results

## 2.3 Movie Effect ##

## 2.3.1 Model ##

################################
# Calculate average rating of movies in tempedx dataset
################################

movie_averages <- tempedx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

################################
# Predict rating of tempvalidation dataset and calculate rmse from movie effect model
################################
  
predicted_ratings <- mu + tempvalidation %>% 
  left_join(movie_averages, by="movieId") %>% 
  pull(b_i)

movieeffect_rmse <- RMSE(predicted_ratings, tempvalidation$rating)
movieeffect_rmse

################################
# Update tibble with the found RMSE
################################

options(pillar.sigfig = 5)
rmse_results <- bind_rows(rmse_results, tibble(method = "Temp Movie Effect", 
                                               RMSE = movieeffect_rmse))
rmse_results

## 2.4 Movie & User Effects ## 

## 2.4.1 Model ##

################################
# Calculate average user rating in tempedx dataset
################################

user_averages <- tempedx %>%
  left_join(movie_averages, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

################################
# Predict rating of tempvalidation dataset and calculate rmse from user and movie effect model
################################

predicted_ratings <- tempvalidation %>% 
  left_join(movie_averages, by="movieId") %>% 
  left_join(user_averages, by="userId") %>% 
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

movieusereffect_rmse <- RMSE(predicted_ratings, tempvalidation$rating)
movieusereffect_rmse

################################
# Update tibble with the found RMSE
################################

options(pillar.sigfig = 5)
rmse_results <- bind_rows(rmse_results, tibble(method = "Temp Movie & User Effects", 
                                               RMSE = movieusereffect_rmse))
rmse_results

## 2.5 Regularized Movie & User Effects ##

################################
# Plot a graph for number of ratings per movie
################################

edx %>% 
  group_by(movieId) %>% 
  summarize(n=n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "red") + 
  scale_x_log10() + 
  ggtitle("Number of Ratings per Movie") + 
  xlab("Number of Ratings") + 
  ylab("Number of Movies")

################################
# Plot a graph for number of ratings per user
################################

edx %>% 
  group_by(userId) %>% 
  summarize(n=n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "red") + 
  scale_x_log10() + 
  ggtitle("Number of Ratings per User") + 
  xlab("Number of Ratings") + 
  ylab("Number of Users")

## 2.5.1 Model ##

################################
# Predict rating of tempvalidation dataset and calculate rmse from reg user and movie effect model
################################

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(tempedx$rating)
  
  b_i <- tempedx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- tempedx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    tempvalidation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, tempvalidation$rating))
})

regmovieusereffect_rmse <- min(rmses)
regmovieusereffect_rmse

################################
# Update tibble with the found RMSE
################################

options(pillar.sigfig = 5)
rmse_results <- bind_rows(rmse_results, tibble(method = "Temp Regularised Movie & User Effects", 
                                               RMSE = regmovieusereffect_rmse))
rmse_results

## 3 Results ##

## 3.1 Modelling Results & Performance

################################
# View RMSE results
################################

options(pillar.sigfig = 5)
rmse_results

## 3.2 Final RMSE Result of Selected Model ##

################################
# Find the best tuning parameter
################################

lambdas[which.min(rmses)]

################################
# Predict movie ratings for validation dataset
################################

mu <- mean(edx$rating)

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+5))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+5))
  
predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

################################
# Viewing head of predicted ratings
################################

head(predicted_ratings)

################################
# Calculate RMSE for predicted ratings of validation dataset
################################

final_rmse <- RMSE(predicted_ratings, validation$rating)
final_rmse

## 4 Conclusion ##

## 4.1 Summary ##

## 4.2 Limitations & Future Work ##

## 5 Operating System ##

################################
# View version of machine
################################

version
