# Bias-variance trade-off



Matlab code implemented in R taken from:
<https://theclevermachine.wordpress.com/2013/04/21/model-selection-underfitting-overfitting-and-the-bias-variance-tradeoff/>

with added bootstrapping. The original article probably makes a mistake in looking only at fixed training and test sets instead of bootstrapping them.

``` {.r}
library(dplyr)
library(tidyr)
library(broom)
library(ggplot2)


# target function f(x)
f <- function(x) sin(pi*x)

# define functions
get_mse <- function(pred,true) mean((pred-true)^2)

# add noise to the data
get_noisy_data <- function(x,sd_noise) f(x)+rnorm(length(x),mean=0,sd=sd_noise)

# split each bootstrap sample into train and test sets
split_train_test <- function(x,sd_noise,p_train) {
   x_train <- x[1:ceiling(p_train*nrow(x)),1]
   y_train <- get_noisy_data(x=x_train,sd_noise=sd_noise)
   x_test <- x[(ceiling(p_train*nrow(x))+1):nrow(x),1]
   y_test <- get_noisy_data(x=x_test,sd_noise=sd_noise)
   train <- data.frame(x=x_train$x,y=y_train$x)
   test <- data.frame(x=x_test$x,y=y_test$x)
   return(list(train=train,test=test))
}

# calculate predicted train and test samples
get_predictions <- function(x,sd_noise,p_train,poly_degree) {
    # create data
    dat <- split_train_test(x,sd_noise,p_train)
    # train model
    mod_train <- lm(data=dat$train,y~poly(x,poly_degree))
    # get predictions
    pred_train  <- predict(mod_train)
    pred_test <- predict(mod_train,newdata=dat$test)
    return(data.frame(split=c(rep("train",length(pred_train)),
                              rep("test",length(pred_test))),
                      prediction=c(pred_train,pred_test),
                      y = c(dat$train$y,dat$test$y),
                      x = c(dat$train$x,dat$test$x)))
}

# based on the bootstrap samples, we could easily calculate confidence intervals, not only the expected values of the KPIs
get_kpis <- function(x,n_boot,sd_noise,p_train,poly_degree) {
    repl <- data.frame(x) %>% bootstrap(n_boot) %>%
             do(get_predictions(.,sd_noise=sd_noise,p_train=p_train,poly_degree=poly_degree))
    errors <- repl %>% group_by(split) %>% summarize(mse=get_mse(pred=prediction,true=y))
    bias_squared <- (repl %>% group_by(x) %>% 
                       summarize(mean_y = mean(prediction)) %>% select(mean_y)-f(x))^2 %>% mean()
    variance <- repl %>% filter(split=="test") %>% 
                summarize(variance=var(prediction)) %>% summarize(variance=mean(variance))
    kpis <- data.frame(test_error=errors$mse[1],
                       train_error=errors$mse[2],
                       bias_squared=bias_squared,
                       variance=variance)
    return(kpis)
}
```

``` {.r}
# -------------- define constants ----------------- #

# number of examples per dataset
n_example <- 25
# number of datasets
n_data <- 100
# std of noise
sd_noise <- .5
# proportion of training examples
p_train <- .9
# maximum model complexity
n_poly_max <- 10
# number of bootstrap repetitions
n_boot <- 1000
# x axis
x <- seq(-1,1,length=n_example)


# -------------------- run ------------------------ #

# calculate kpis
kpis <- data.frame(poly_degree=1:n_poly_max,
                   t(sapply(FUN=get_kpis,x=x,n_boot=n_boot,sd_noise=.5,p_train=.9,X=1:n_poly_max)))
# cast to long format
kpi_clean <- data.frame(gather(kpis,kpi,value,-poly_degree))

# ----------------- plot -------------------------- #
ggplot(data=kpi_clean,aes(x=poly_degree,y=unlist(value),color=kpi))+geom_line()+scale_y_continuous("value")+scale_x_continuous("complexity",breaks=1:n_poly_max)
```

![bias_variance_trade_off](https://github.com/adamlenart/MIDS-W261/blob/master/bias_variance/images/bias_variance.png)
