library(tidymodels)
library(vroom)
library(embed)
library(ranger)

ggg_train <- vroom("train.csv")
ggg_test <- vroom("test.csv")

ggg_recipe <- recipe(type ~ ., data = ggg_train) |>
  step_novel(all_nominal_predictors()) |>
  step_other(all_nominal_predictors(), threshold = 0.01) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

ggg_recipe2 <- recipe(type ~ ., data = ggg_train) |>
  step_rm(id) |>
  step_mutate(color) |>
  step_dummy(factor)


prep <- prep(ggg_recipe2)
baked <- bake(prep, new_data = ggg_train)
dim(baked)

rf_mod <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 1000
) |>
  set_engine("ranger") |>
  set_mode("classification")

rf_wf <- workflow() |>
  add_recipe(ggg_recipe) |>
  add_model(rf_mod)

folds <- vfold_cv(ggg_train, v = 5, strata = type)

rf_grid <- grid_regular(mtry(range(1, 10)),
                        min_n(),
                        levels = 5)

CV_results_rf <- rf_wf |>
  tune_grid(resamples = folds_rf,
            grid = rf_grid,
            metrics = metric_set(accuracy))

rf_best <- CV_results_rf |>
  select_best(metric = "accuracy")

rf_final <- finalize_workflow(rf_wf, rf_best) |>
  fit(data = ggg_train)

rf_predictions <- rf_final |>
  predict(new_data = ggg_test, type = "class") |>
  bind_cols(ggg_test) |>
  rename(type = .pred_class) |>
  select(id, type)

vroom_write(rf_predictions, "./RFPreds.csv", delim = ",")


library(discrim)
library(naivebayes)

nb_mod <- naive_Bayes(Laplace = tune(),
                      smoothness = tune()) |>
  set_mode("classification") |>
  set_engine("naivebayes")

nb_wf <- workflow() |>
  add_recipe(ggg_recipe) |>
  add_model(nb_mod)

folds_nb <- vfold_cv(ggg_train, v = 5, strata = type)

nb_grid <- grid_regular(Laplace(),
                        smoothness(),
                        levels = 3)

CV_results_nb <- nb_wf |>
  tune_grid(resamples = folds_nb,
            grid = nb_grid,
            metrics = metric_set(accuracy))

nb_best <- CV_results_nb |>
  select_best(metric = "accuracy")

nb_final <- finalize_workflow(nb_wf, nb_best) |>
  fit(data = ggg_train)

nb_predictions <- nb_final |>
  predict(new_data = ggg_test, type = "class") |>
  bind_cols(ggg_test) |>
  rename(type = .pred_class) |>
  select(id, type)

vroom_write(nb_predictions, "./NBPreds.csv", delim = ",")
