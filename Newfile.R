
train <- read.csv("train.csv", stringsAsFactors=FALSE)
test  <- read.csv("test.csv",  stringsAsFactors=FALSE)

test$Survived <- NA

modtitanic <-rbind(train, test)

Names <- modtitanic$Name
Title <-  gsub("^.*, (.*?)\\..*$", "\\1", Names)

modtitanic$Title <- Title
table(modtitanic$Sex,modtitanic$Title)

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

modtitanic$Title[modtitanic$Title == 'Mlle']        <- 'Miss' 
modtitanic$Title[modtitanic$Title == 'Ms']          <- 'Miss'
modtitanic$Title[modtitanic$Title == 'Mme']         <- 'Mrs' 
modtitanic$Title[modtitanic$Title %in% rare_title]  <- 'Rare Title'
table(modtitanic$Sex, modtitanic$Title)


modtitanic$FamilySize <-modtitanic$SibSp + modtitanic$Parch + 1

modtitanic$FamilySized[modtitanic$FamilySize == 1]   <- 'Single'
modtitanic$FamilySized[modtitanic$FamilySize < 5 & modtitanic$FamilySize >= 2]   <- 'Small (2-4)'
modtitanic$FamilySized[modtitanic$FamilySize >= 5]   <- 'Big (>=5)'


ticket.unique <- rep(0, nrow(modtitanic))
tickets <- unique(modtitanic$Ticket)
for (i in 1:length(tickets)) {
  current.ticket <- tickets[i]
  party.indexes <- which(modtitanic$Ticket == current.ticket)
  for (k in 1:length(party.indexes)) {
    ticket.unique[party.indexes[k]] <- length(party.indexes)
  }
}
modtitanic$ticket.unique <- ticket.unique
modtitanic$TicketSize[modtitanic$ticket.unique == 1]   <- 'Single'
modtitanic$TicketSize[modtitanic$ticket.unique < 5 & modtitanic$ticket.unique>= 2]   <- 'Small (2-4)'
modtitanic$TicketSize[modtitanic$ticket.unique >= 5]   <- 'Big (>=5)'


modtitanic$IsSolo[modtitanic$SibSp==0] <- 'Yes'
modtitanic$IsSolo[modtitanic$SibSp!=0] <- 'No'
modtitanic$IsSolo <- as.factor(modtitanic$IsSolo)

modtitanic$Embarked[modtitanic$Embarked==""]="S"

factor_vars <- c('PassengerId','Pclass','Sex','Embarked','Title','Surname', 'FamilySized', 'TicketSize' , 'IsSolo')

set.seed(123)
mice_mod <- mice(modtitanic[, !names(modtitanic) %in% c('PassengerId','Name','Ticket',
                                                            'Cabin','Family','Surname','Survived')], method='rf')

mice_output <- complete(mice_mod)

modtitanic$Age <- mice_output$Age
sum(is.na(modtitanic$Age))

modtitanic$Child[modtitanic$Age < 18] <- 'Child (<18)'
modtitanic$Child[modtitanic$Age >= 18] <- 'Adult (>=18)'
table(modtitanic$Child, modtitanic$Survived)

modtitanic$Mother <- 'Not Mother'
modtitanic$Mother[modtitanic$Sex == 'female' & modtitanic$Parch > 0 & modtitanic$Age > 18 & modtitanic$Title != 'Miss'] <- 'Mother'
table(modtitanic$Mother, modtitanic$Survived)

modtitanic$Child  <- factor(modtitanic$Child)
modtitanic$Mother <- factor(modtitanic$Mother)
str(modtitanic)

modtitanic$Survived<- as.factor(modtitanic$Survived)
modtitanic$Sex<- as.factor(modtitanic$Sex)
modtitanic$Embarked<- as.factor(modtitanic$Embarked)
modtitanic$Title <- as.factor(modtitanic$Title)
modtitanic$FamilySized <- as.factor(modtitanic$FamilySized)
modtitanic$TicketSize <- as.factor(modtitanic$TicketSize)
modtitanic$IsSolo<- as.factor(modtitanic$IsSolo)
modtitanic$Child<- as.factor(modtitanic$Child)
modtitanic$Mother<- as.factor(modtitanic$Mother)
modtitanic$Pclass<- as.factor(modtitanic$Pclass)


train<-modtitanic[1:891, c('Survived','Sex','Embarked','Title','FamilySized','IsSolo', 
                             'Child', 'Mother', 'Pclass')]
set.seed(12345)
ind<-createDataPartition(train$Survived,times=1,p=0.8,list=FALSE)

train_80=train[ind,]
test_20=train[-ind,]

str(ind)

round(prop.table(table(train$Survived)*100),digits = 1)

                            ###########    Random forest   ##############
set.seed(12345)
model_rf<- randomForest(x = train_80[,-1],y=train_80[,1], importance = TRUE, ntree = 1000)
model_rf
varImpPlot(model_rf)
pred_rf2=predict(model_rf,newdata = test_20)
confusionMatrix(pred_rf2,test_20$Survived)

                            ################   SVM   #######################
set.seed(12345)
model_svm=tune.svm(Survived~.,data=train_80,kernel="linear",cost=c(1000))
model_svm
best_linear=model_svm$best.model
best_linear
pred_test=predict(best_linear,newdata=test_20,type="class")
summary(pred_test)
confusionMatrix(pred_test,test_20$Survived)

                          ############ Gradient Boosting Model #############

train <- read.csv("train.csv", stringsAsFactors=FALSE)
test  <- read.csv("test.csv",  stringsAsFactors=FALSE)

test$Survived <- rep(NA, nrow(test))
comb   <- rbind(train,test)
train_idx = which(!is.na(modtitanic$Survived))

trControl <- trainControl(method="repeatedcv", number=5, repeats=2);
xgbGrid <- expand.grid(nrounds=c(20),
                       max_depth=c(3),
                       eta=c(0.1),
                       colsample_bytree=c(0.2,0.8),
                       subsample=c(0.2,0.8),
                       gamma=c(0),
                       min_child_weight=c(1))

model_xgb <- train(factor(Survived) ~ Sex + Pclass + Embarked, data = comb[train_idx,], 
                   trControl=trControl, method='xgbTree', tuneGrid = xgbGrid);
print(model_xgb)
pred <- predict(model_xgb, comb[-train_idx,])
submit <- data.frame(PassengerId = test$PassengerId, Survived = pred);
write.csv(submit, file = "mysubmission.csv", row.names = FALSE, quote = FALSE)

