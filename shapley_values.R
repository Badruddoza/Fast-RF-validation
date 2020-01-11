# SHAP measures the impact of variables taking into account 
# the interaction with other variables.
# The shap value table changes the original data so has the same dimension.
# The Shapley value is the average marginal contribution of a feature value 
# across all possible coalitions.
# Given the current set of feature values, the contribution of a feature value 
# to the difference between the actual prediction and the mean prediction is 
# the estimated Shapley value.
rm(list=ls())
data("Boston", package  = "MASS")
head(Boston)
set.seed(42)
#install.packages("iml")
library("iml")
library("randomForest")
data("Boston", package  = "MASS")
rf = randomForest(medv ~ ., data = Boston, ntree = 50)

# using the imp Predictor
X = Boston[which(names(Boston) != "medv")]
predictor = Predictor$new(rf, data = X, y = Boston$medv)

# Feature importance
imp = FeatureImp$new(predictor, loss = "mae")
plot(imp)

# Feature effects
ale = FeatureEffect$new(predictor, feature = "lstat")
ale$plot()

# Measure feature interactions
interact = Interaction$new(predictor)
plot(interact)
# (two way interactions)
interact = Interaction$new(predictor, feature = "crim")
plot(interact)
# (plot all feature effects)
effs=FeatureEffects$new(predictor)
plot(effs)

# Plot a surrogate model tree
tree=TreeSurrogate$new(predictor,maxdepth=2)
plot(tree)

# Use the tree to make predictions
head(tree$predict(Boston))

# Explain single predictors with a local model using LIME
lime.explain=LocalModel$new(predictor, x.interest=X[1,])
lime.explain$results
plot(lime.explain)
lime.explain$explain(X[2,])
plot(lime.explain)

# Explain single predictions with game theory (Shapley values)
# X are players, y is payout. The Shapley value fairly distributes the payout.
shapley=Shapley$new(predictor, x.interest=X[1,])
shapley$plot()
# (use a different row e.g. the second row)
shapley$explain(x.interest=X[2,])
shapley$plot()
# (find the results)
results=shapley$results
head(results)
