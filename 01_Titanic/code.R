# tbj2cu

# Set the working directory and read in the data
setwd("F:/2018 Fall/SYS 6018/assignments/kaggle/01_Titanic")
tr = read.csv("train.csv",header=TRUE)

# Determine what ratio to split the training data into training/validation sets
# by calculating the standard deviation of the Bernoulli variable as followed.
# Estimate the values of p and q by calculating the survival rate.
p = sum(tr$Survived==1)/length(tr$Survived)
q = 1-p
# Very roughly we have a ratio of 0.3838
alp = 0.05
n = p*q/alp**2
# If these assumptions hold, we should have n>=95 to have a standard error of
# less than 0.05. 

# The maxima is given at p=0.5000 where n=100
p = .5
q = 1-p
alp = 0.05
n = p*q/alp**2

# To be as generous as possible set n=100

# Let's clean up the data a little.

# First, there are a few missing age entries. There is not much that can
# be done to recover those missing values.

# Second, the ticket number is odd. I'm not sure what to make of it---it is
# written in different formats and there are duplicate values.
tr = tr[,!(names(tr) %in% c("Ticket"))]

# Third, some people have several cabins. Extract the floor where each of
# these cabins were by removing numerical values and taking the first
# character of the resulting string.
temp = tr[tr$Cabin!="",c("PassengerId","Survived","Cabin")]
tempcol = gsub('[[:digit:]]+',"",temp$Cabin)
col = substr(tempcol, start=1, stop=1)
temp$CabinLevel = col
aggregate(Survived~CabinLevel, temp, mean)
temp = temp[,!names(temp) %in% c("Cabin", "Survived")]

# Merge
tr = tr[,!(names(tr) %in% c("Cabin"))]
tr = merge(tr, temp, by="PassengerId", all.x=TRUE)

# Let's see some statistics
aggregate(Survived~Pclass, tr, mean)
#   Pclass  Survived
# 1      1 0.6296296
# 2      2 0.4728261
# 3      3 0.2423625

aggregate(Survived~Sex, tr, mean)
#      Sex  Survived
# 1 female 0.7420382
# 2   male 0.1889081

agegroup = aggregate(Survived~Age, tr, mean)
plot(agegroup$Age, agegroup$Survived)

aggregate(Survived~SibSp, tr, mean)
#   SibSp  Survived
# 1     0 0.3453947
# 2     1 0.5358852
# 3     2 0.4642857
# 4     3 0.2500000
# 5     4 0.1666667
# 6     5 0.0000000
# 7     8 0.0000000

aggregate(Survived~Parch, tr, mean)
#   Parch  Survived
# 1     0 0.3436578
# 2     1 0.5508475
# 3     2 0.5000000
# 4     3 0.6000000
# 5     4 0.0000000
# 6     5 0.2000000
# 7     6 0.0000000

# Round to the nearest dollar to simulate 'bins'
faregroup = aggregate(Survived~round(Fare), tr, mean)
plot(faregroup$'round(Fare)', faregroup$Survived)
# It's a little clearer that there was a high survival rate for those who
# bought tickets in the 100 dollar range (cluster).

aggregate(Survived~CabinLevel, tr, mean)
#   CabinLevel  Survived
# 1          A 0.4666667
# 2          B 0.7446809
# 3          C 0.5932203
# 4          D 0.7575758
# 5          E 0.7500000
# 6          F 0.6153846
# 7          G 0.5000000
# 8          T 0.0000000

aggregate(Survived~Embarked, tr, mean)
#   Embarked  Survived
# 1          1.0000000
# 2        C 0.5535714
# 3        Q 0.3896104
# 4        S 0.3369565

# The passengers 62 and 830 are outliers without an embark location.

apply(tr, 2, is.factor)
tr$Survived = as.factor(tr$Survived)
tr$Pclass = as.factor(tr$Pclass)
tr$Sex = as.factor(tr$Sex)
tr$Embarked = as.factor(tr$Embarked)
tr$CabinLevel = as.factor(tr$CabinLevel)

# Method 1: Logistic Regression

# Logistic regression doesn't like empty values. Either remove rows with missing
# values (assuming Age is missing randomly) or impute using the mean. I don't
# think that using the mean would be benefitial as there is a wide range of ages
# on the list of ages. Instead, just remove the missing rows.
sum(is.na(tr$Age))
train = tr[is.na(tr$Age)==0,]

# same with CabinLevel, there's not much we can do but remove the entire column
train = train[,!names(train) %in% c("CabinLevel")]

# Fit the model
full = glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked, data=train, family=binomial)
null = glm(Survived~1, data=train, family=binomial)

# Variable selection
fo = step(null, scope=list(lower=null, upper=full), direction="forward")
ba = step(full, data=train, direction="backward")
bo = step(null, scope=list(upper=full), data=train, direction="both")

formula(fo)
# Survived ~ Sex + Pclass + Age + SibSp
formula(ba)
# Survived ~ Pclass + Sex + Age + SibSp
formula(bo)
# Survived ~ Sex + Pclass + Age + SibSp

# This is a unanimous conclusion. We will use these variables for calculating
# the final model.

model.logreg = glm(Survived~Sex+Pclass+Age+SibSp, data=train, family=binomial)
summary(model.logreg)

# Read in the test data
test = read.csv("test.csv",header=TRUE)
test$Pclass = as.factor(test$Pclass)
test$Sex = as.factor(test$Sex)
test$Embarked = as.factor(test$Embarked)

# For the predictions we HAVE to have complete rows. The easiest way is to use
# the simple mean or median. We will use the median because of the wide range
# of age values

medianage = median(test$Age[is.na(test$Age)==0])
test$Age[is.na(test$Age)==1] = medianage

# Same for fare there is one missing value

medianfare = median(test$Fare[is.na(test$Fare)==0])
test$Fare[is.na(test$Fare)==1] = medianfare

pred = predict(model.logreg, newdata=test, type = "response")
out = cbind(test$PassengerId, pred>=0.5)
colnames(out) = c("PassengerId", "Survived")
write.csv(out, file="titanic_survival_predictions.csv", row.names=FALSE)

# This scored a 0.74162 not bad, but lots of improvements can be made.
# Let's try (for fun) what accuracy we can get by using the full model

pred = predict(full, newdata=test, type = "response")
out = cbind(test$PassengerId, pred>=0.5)
colnames(out) = c("PassengerId", "Survived")
write.csv(out, file="titanic_survival_predictions_full.csv", row.names=FALSE)

# This scored 0.75119 BUT is probably vulnerable to overfitting
# Let's try using something other than logistic regression. How about...

# Method 2: Random Forests

install.packages("randomForest"); library(randomForest)

x = tr[, names(tr) %in% c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","CabinLevel")]
y = tr[, names(tr) %in% c("Survived")]
x[x$Embarked=="",]$Embarked = NA
# Automatic imputation
imp = rfImpute(x,y,iter=5,ntree=5000)
imp$Embarked = factor(imp$Embarked)
colnames(imp)[1] = "Survived"
colnames(imp)[9] = "CabinLevel"

temp = test[test$Cabin!="",c("PassengerId","Cabin")]
tempcol = gsub('[[:digit:]]+',"",temp$Cabin)
col = substr(tempcol, start=1, stop=1)
temp$CabinLevel = col
temp = temp[,!names(temp) %in% c("Cabin")]

test = read.csv("test.csv",header=TRUE)
test$Pclass = as.factor(test$Pclass)
test$Sex = as.factor(test$Sex)
test$Embarked = as.factor(test$Embarked)

test = test[,!(names(test) %in% c("Cabin"))]
test = merge(test, temp, by="PassengerId", all.x=TRUE)

test = test[,names(test) %in% c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","CabinLevel")]
test$CabinLevel = factor(test$CabinLevel)
levels(test$CabinLevel) = c(levels(test$CabinLevel),"T")

test2 = rfImpute(test[,!names(test) %in% c("Pclass")],test[,names(test) %in% c("Pclass")],iter=5,ntree=5000)
colnames(test2)[1] = "Pclass"

model.forest = randomForest(Survived~., data=imp, ntree=10000)
preds=predict(model.forest, test2)

out = data.frame(out[,1], preds)
colnames(out) = c("PassengerId", "Survived")
write.csv(out, file="titanic_survival_predictions_tree1.csv", row.names=FALSE)

# This was not an improvement at all.

# Method 3: Random Forests (again)

# This time let's use all of the significant factors found in the first part

x = tr[, names(tr) %in% c("Pclass","Sex","Age","SibSp")]
y = tr[, names(tr) %in% c("Survived")]
imp = rfImpute(x,y,iter=5,ntree=5000)
colnames(imp)[1] = "Survived"
model.forest = randomForest(Survived~., data=imp, ntree=25000)
preds=predict(model.forest, test)
out = data.frame(out[,1], preds)
colnames(out) = c("PassengerId", "Survived")
write.csv(out, file="titanic_survival_predictions_tree2.csv", row.names=FALSE)

# This pushes the score up to a new high at 0.77511 and also better avoids
# overfitting.

# Method 4: Random Forests (again and again but use more variables)
# Note this may not be a good idea, but let's test it

x = tr[, names(tr) %in% c("Pclass","Sex","Age","SibSp", "Parch", "Fare", "Embarked")]
y = tr[, names(tr) %in% c("Survived")]
imp = rfImpute(x,y,iter=5,ntree=5000)
colnames(imp)[1] = "Survived"
model.forest = randomForest(Survived~., data=imp, ntree=25000)
preds=predict(model.forest, test)
out = data.frame(out[,1], preds)
colnames(out) = c("PassengerId", "Survived")
write.csv(out, file="titanic_survival_predictions_tree3.csv", row.names=FALSE)

# As expected this only scored 0.77511 which is on par with the previous model

