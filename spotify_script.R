#LOAD NECESSARY PACKAGES
library(spotifyr) #API interaction
library(tidyverse) #cleaning data, ggplot, etc 
library(tidymodels) #for modeling/statistical analysis
library(rsample) #for splitting data into train / test
library(recipes) #for creating the recipe for ML
library(kknn) #for KNN modeling
library(plotly) #for data viz
library(ggpubr) #for data viz
library(here) #for simplifying file path navigation
library(baguette) #for bagging decision trees
library(ranger) # engine for random forests
library(kableExtra) #for creating a formatted table
library(DT)



#GETTING A DATAFRAME OF LIKED SONGS

#In the get_my_saved_tracks() function, you can use offsets() to specify the index of the first track to return. 
offsets = seq(from = 0, to = 1000, by = 50)

#initializing an empty matrix 
liked_tracks <- data.frame(matrix(nrow = 0, ncol = 30))

#Function to get my 1050 most recently liked tracks 
for (i in seq_along(offsets)) {  
  my_tracks = get_my_saved_tracks(limit = 50, 
                                  offset = offsets[i])
  df_temp = as.data.frame(my_tracks) #creating a temporary data frame to store the 50 liked tracks from a given run
  liked_tracks <- rbind(liked_tracks, df_temp) #binding the temporary data frame to my liked tracks data frame. 
}




#obtain a list of the track IDs 
ids <- liked_tracks$track.id

#the ids argument in `get_track_audio_features()` can only take 100 IDs at a time, so I'm splitting them into 100 track groupings
ids_split <- split(ids, ceiling(seq_along(ids) / 100))

#create an empty list
my_tracks_audio_feats_list <- list()

#Iterating the `get_track_audio_features()`function over each ID split and storing the resulting data in a list.
for (i in 1:length(ids_split)) {
  my_tracks_audio_feats_list[[i]] <- get_track_audio_features(ids = ids_split[[i]])
}

#Combine the list of data frames into a single data frame.
my_tracks_audio_feats <- do.call(rbind, my_tracks_audio_feats_list)




#selecting the track id and track name from liked tracks so I can use left_join and only add the track name to the audio features data set
liked_tracks_join <- liked_tracks %>%
  select(track.id, track.name, track.artists) %>%
  mutate(primary_artist = unlist(lapply(liked_tracks$track.artists, function(x) x$name[1]))) %>%
  select(-track.artists)

#add the track name and artist to the audio features data frame 
liked_tracks_full <- left_join(my_tracks_audio_feats, liked_tracks_join, by = c("id" = "track.id"))




# reading in Elke's liked tracks. Add a column to make it clear which tracks are Elke's  
elke_liked_tracks <- read_csv("/Users/lewiswhite/MEDS/eds-232-ML/Machine-Learning/elke_liked_tracks.csv") %>%
  mutate(listener = "Elke")

#add a column named "listener" to make it clear that these are my tracks 
lewis_liked_tracks <- liked_tracks_full %>%
  mutate(listener = "Lewis")

# Combine all of our liked tracks into one data frame 
all_tracks <- rbind(lewis_liked_tracks, elke_liked_tracks) %>%
  select(-(type:analysis_url))



#Danceability comparison
dance_plot <- ggplot(all_tracks, aes(x = danceability, fill = listener,
                                     text = paste(listener))) +
  geom_density(alpha=0.6, color=NA) +
  scale_fill_manual(values=c("#b0484f", "#4436d9"))+
  labs(x="Danceability", y="Density") +
  guides(fill=guide_legend(title="Listener"))+
  theme_minimal() +
  ggtitle("Distribution of Danceability Data")


#speechiness comparison
speech_plot <- ggplot(all_tracks, aes(x = speechiness, fill = listener,
                                      text = paste(listener))) +
  geom_density(alpha=0.6, color=NA) +
  scale_fill_manual(values=c("#b0484f", "#4436d9"))+
  labs(x="Speechiness", y="Density") +
  guides(fill=guide_legend(title="Listener"))+
  theme_minimal() +
  ggtitle("Distribution of Speechiness Data")


#acousticness comparison
acoustic_plot <- ggplot(all_tracks, aes(x = acousticness, fill = listener,
                                        text = paste(listener))) +
  geom_density(alpha=0.6, color=NA) +
  scale_fill_manual(values=c("#b0484f", "#4436d9"))+
  labs(x="Acousticness", y="Density") +
  guides(fill=guide_legend(title="Listener"))+
  theme_minimal() +
  ggtitle("Distribution of Acousticness Data")

#energy comparison 
energy_plot <- ggplot(all_tracks, aes(x = energy, fill = listener,
                                      text = paste(listener))) +
  geom_density(alpha=0.6, color=NA) +
  scale_fill_manual(values=c("#b0484f", "#4436d9"))+
  labs(x="Energy", y="Density") +
  guides(fill=guide_legend(title="Listener"))+
  theme_minimal() +
  ggtitle("Distribution of Energy Data")

ggarrange(dance_plot, speech_plot, acoustic_plot, energy_plot, ncol=2, nrow=2, common.legend = TRUE, legend="bottom")

#My music appears to be higher energy, more danceable, and more "speechy," while Elke's music tends to be more acoustic. 




#prepare all_tracks for modeling by removing songs with duplicate values (tough to predict if both Elke and I liked a song), making nominal data a factor, and removing variables that don't make sense to include in the model
all_tracks_modeling <- all_tracks[!duplicated(all_tracks$track.name, fromLast = TRUE) & !duplicated(all_tracks$track.name), ] %>%  
  mutate_if(is.ordered, .funs = factor, ordered = F) %>%
  select(-track.name, -primary_artist) %>%
  mutate(listener = as.factor(listener))





#splitting the data
set.seed(123)
#initial split of data ~ we're going with 70/30 because the sample size isn't super large for testing
tracks_split <- initial_split(all_tracks_modeling, prop = .7)
tracks_train <- training(tracks_split)
tracks_test <- testing(tracks_split)

#pre-processing
# We need to create a recipe and do the pre-processing by converting dummy coding the nominal variables and normalizing the numeric variables.
tracks_recipe <- recipe(listener ~ ., data = tracks_train) %>%
  #step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>% #normalize numeric to make sure scale is okay
  prep()





# Define our KNN model with tuning
knn_spec_tune <- nearest_neighbor(neighbor = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")


# Define a new workflow
wf_knn_tune <- workflow() %>%
  add_model(knn_spec_tune) %>%
  add_recipe(tracks_recipe)

# 10-fold CV on the training dataset
set.seed(123)
cv_folds <- tracks_train %>%
  vfold_cv(v = 10)


# Fit the workflow on our predefined folds and hyperparameters
fit_knn_cv <- wf_knn_tune %>%
  tune_grid(
    cv_folds, 
    grid = data.frame(neighbors = c(1, 5, seq(10, 150, 10))) # try with 1 nearest neighbor, try with 5, 10, 20, 30, ..., 100
  )



final_wf <-
  wf_knn_tune %>%
  finalize_workflow(select_best(fit_knn_cv))

# Fitting our final workflow
final_fit <- final_wf %>%
  fit(data = tracks_train)

#generating predictions using the model on the test data
tracks_pred <- final_fit %>% predict(new_data = tracks_test)


# Write over 'final_fit' with this last_fit() approach
final_fit <- final_wf %>% last_fit(tracks_split)







#new spec, tell the model that we are tuning hyperparams
tree_spec_tune <- decision_tree(
  cost_complexity = tune(),   #tune() asks R to try a bunch of different parameters. 
  tree_depth = tune(),
  min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

#create a grid of options for tuning purposes so we can identify the optimal value of hyperparameters 

tree_grid <- grid_regular(cost_complexity(), tree_depth(), min_n(), levels = 5) #grid_regular shows grid of different input tuning options while levels says how many combinations we should try. 




wf_tree_tune <- workflow() %>%
  add_recipe(tracks_recipe) %>%
  add_model(tree_spec_tune)

#when you add recipe into workflow, it automatically will prep and bake when necessary. 


#workflow pulls together the specification. and then we can fit on the workflow. you could fit the wf_tree_tune 

#set up k-fold cv. This can be used for all the algorithms. I switch to 5 fold CV instead of 10 (used in KNN above) for computation speed. 
listener_cv <- tracks_train %>% vfold_cv(v=5) #creating 5 folds cross validation 



doParallel::registerDoParallel() #build trees in parallel
#200s

#get the results of the tuning 
tree_rs <- tune_grid(
  wf_tree_tune,
  listener ~ ., #function 
  resamples = listener_cv, #resamples to use
  grid = tree_grid, #grid to try
  metrics = metric_set(accuracy) #how to assess which combinations are best 
)


#picking the best model for the final model
final_tree <- finalize_model(tree_spec_tune, select_best(tree_rs))


#Fit the model on the test data 
final_tree_fit <- last_fit(final_tree, listener ~., tracks_split)





set.seed(123)
bag_spec_tune <- bag_tree(cost_complexity = tune(),
                          tree_depth = tune(),
                          min_n = tune()) %>%
  set_mode("classification") %>%
  set_engine("rpart", times = 50) #50 trees in a bag

#create tuning grid for hyperparamters 
bag_grid <- grid_regular(cost_complexity(), tree_depth(), min_n(), levels = 5)

#make workflow for bagging
wf_bag_tune <- workflow() %>% 
  add_recipe(tracks_recipe) %>%
  add_model(bag_spec_tune)

doParallel::registerDoParallel() #build trees in parallel
#200s

#find results of the tuning process 
bag_rs <- tune_grid(
  bag_spec_tune,
  listener ~ ., #function 
  resamples = listener_cv, #resamples to use
  grid = bag_grid,
  metrics = metric_set(accuracy)) #how to assess which combinations are best 
  
#picking the best model for the final model
final_bag <- finalize_model(bag_spec_tune, select_best(bag_rs))

#fitting the best model on the test data
final_bag_fit <- last_fit(final_bag, listener ~., tracks_split)







set.seed(123)

#setting up the random forest specification
forest_spec_tune <- 
  rand_forest(min_n = tune(),
              mtry = tune(),
              trees =  140) %>% # 14 predictors * 10, as suggested by "Hands on Machine Learning in R" by Bradley Boehmke & Brandon Greenwell. Tuning this value led to an extremely long computation time.
  set_engine("ranger") %>%
  set_mode("classification")

#create grid for tuning min_n and the mtry value
forest_grid <- grid_regular(mtry(c(1,13)), min_n(), levels = 5)

#create a workflow for the random forests
wf_forest_tune <- workflow() %>% 
  add_recipe(tracks_recipe) %>%
  add_model(forest_spec_tune)

doParallel::registerDoParallel() #build forest in parallel

#get results of the tuning to try and find optimal model
forest_rs <- tune_grid(
  forest_spec_tune,
  listener ~ ., 
  resamples = listener_cv, #resamples to use
  grid = forest_grid,
  metrics = metric_set(accuracy) #how to assess which combinations are best 
)


#picking the best model for the final model
final_forest <- finalize_model(forest_spec_tune, select_best(forest_rs))

#fitting the optimal model on the test data
final_forest_fit <- last_fit(final_forest, listener ~., tracks_split)



#cleaning model metrics to create a table with each model's performance metrics 

forest_results_table <- final_forest_fit %>% collect_metrics() %>%
  select(.metric, .estimate) %>%
  mutate(method = "Random Forest")

bagging_results_table <- final_bag_fit %>% collect_metrics() %>%
  select(.metric, .estimate) %>%
  mutate(method = "Bagging")

decision_tree_results_table <- final_tree_fit %>% collect_metrics() %>%
  select(.metric, .estimate) %>%
  mutate(method = "Decision Tree")

knn_results_table <- final_fit %>% collect_metrics() %>%
  select(.metric, .estimate) %>%
  mutate(method = "KNN")

majority_class_results_table <- data.frame (.metric  = c("accuracy", "roc_auc"),
                                            .estimate = c(nrow(lewis_liked_tracks) / (nrow(lewis_liked_tracks) + nrow(elke_liked_tracks)), .5),
                                            method = c("Dummy Classifier", "Dummy Classifier")
)

full_results <- bind_rows(majority_class_results_table,
                          knn_results_table, 
                          decision_tree_results_table,
                          bagging_results_table,
                          forest_results_table) %>%
  select(method, .estimate, .metric)
