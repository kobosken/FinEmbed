#Install necessary packages If not done before:
install.packages("lubridate")
install.packages("riskParityPortfolio")
install.packages("NMOF")

#### Initialization, Data Import, log return calcs, and important dates####
#Set Working Directory & necessary packages
setwd("~/Desktop/1NCode")
library(lubridate) #For date calcs
library(riskParityPortfolio) #For Risk Parity
library(NMOF) #For Min Var

#df will be the dataframe with all raw data
df<-read.csv("US_set.csv")

#As the first col is just numbers, remove, and turn dates into R date format
df<-df[,c(2:length(df))]
df$Date<-as.Date(df$Date,"%Y-%m-%d")

#As log returns are n*m-1 dimensions, we initial a matrix of that size and perform the calcs for log returns
dflr<-df[c(2:dim(df)[1]),]
for(i in c(1:(length(df)-1))){
  dflr[,i+1]<-diff(log(df[,i+1]), lag=1)
}

#Option to view output
View(dflr)
rownames(dflr) <- 1:nrow(dflr)

#Find and store the index of the first date available in every month (as it'll be relevant for covariance matirx calcs)
cutoff_dates<-diff(floor_date(dflr$Date, "month"))>1
cov_dates<-which(cutoff_dates)+1
cov_dates<-c(1,cov_dates)



#### Computing covariance and portfolio weights ####

#Calculate and store each covariance matrix which consists of 3 months of data rolling/shifting monthly
#Note: index starts at "4" - first date where we have enough data
covmatrix<-list()
for(i in c(4:(length(cov_dates)))){
  covmatrix[[i]]<-cor(dflr[c(cov_dates[i-3]:(cov_dates[i]-1)),c(2:dim(dflr)[2])])
}

#1 Over N Portfolio:
weights_1n_single_period<-rep(1/(dim(dflr)[2]-1),(dim(dflr)[2]-1))
w_1n<-list()
for(i in c(4:(length(cov_dates)))){
  w_1n[[i]]<-weights_1n_single_period
}

#Risk Parity:

w_rp<-list()
for(i in c(4:(length(cov_dates)))){
  w_rp[[i]]<-as.numeric(riskParityPortfolio(covmatrix[[i]])$w)
}

#Min Variance:

w_mv<-list()
for(i in c(4:(length(cov_dates)))){
  w_mv[[i]]<-round(as.numeric(minvar(as.matrix(covmatrix[[i]]), wmin = 0.000000001, wmax = 1, method = "qp")), digits = 5)
}

#### Performance of Portfolios ####

#Initialize vectors to store daily portfolio performance
daily_perform_1n<-rep(NA, nrow(dflr))
daily_perform_rp<-rep(NA, nrow(dflr))
daily_perform_mv<-rep(NA, nrow(dflr))

#Calc vectors of daily portfolio performance
for(i in c(4:(length(cov_dates)))){
  daily_perform_1n[c(cov_dates[i-1]:(cov_dates[i]-1))]<-rowSums(sweep(dflr[c(cov_dates[i-1]:(cov_dates[i]-1)),
                                                                           c(2:dim(dflr)[2])], MARGIN=2, w_1n[[i]], `*`))
}

for(i in c(4:(length(cov_dates)))){
  daily_perform_rp[c(cov_dates[i-1]:(cov_dates[i]-1))]<-rowSums(sweep(dflr[c(cov_dates[i-1]:(cov_dates[i]-1)),
                                                                           c(2:dim(dflr)[2])], MARGIN=2, w_rp[[i]], `*`))
}

for(i in c(4:(length(cov_dates)))){
  daily_perform_mv[c(cov_dates[i-1]:(cov_dates[i]-1))]<-rowSums(sweep(dflr[c(cov_dates[i-1]:(cov_dates[i]-1)),
                                                                           c(2:dim(dflr)[2])], MARGIN=2, w_mv[[i]], `*`))
}

#Find summary stats:
#Vols
monthly_vol_1n<-rep(NA,length(cov_dates))
monthly_vol_rp<-rep(NA,length(cov_dates))
monthly_vol_mv<-rep(NA,length(cov_dates))
montly_days_in_month<-rep(NA,length(cov_dates))

for(i in c(4:(length(cov_dates)))){
  monthly_vol_1n[i]<-sd(daily_perform_1n[c(cov_dates[i-1]:(cov_dates[i]-1))])
}
for(i in c(4:(length(cov_dates)))){
  monthly_vol_rp[i]<-sd(daily_perform_rp[c(cov_dates[i-1]:(cov_dates[i]-1))])
}
for(i in c(4:(length(cov_dates)))){
  monthly_vol_mv[i]<-sd(daily_perform_mv[c(cov_dates[i-1]:(cov_dates[i]-1))])
}
for(i in c(4:(length(cov_dates)))){
  montly_days_in_month[i]<-length(c(cov_dates[i-1]:(cov_dates[i]-1)))
}

#returns
monthly_r_1n<-rep(NA,length(cov_dates))
monthly_r_rp<-rep(NA,length(cov_dates))
monthly_r_mv<-rep(NA,length(cov_dates))

for(i in c(4:(length(cov_dates)))){
  monthly_r_1n[i]<-sum(daily_perform_1n[c(cov_dates[i-1]:(cov_dates[i]-1))])
}
for(i in c(4:(length(cov_dates)))){
  monthly_r_rp[i]<-sum(daily_perform_rp[c(cov_dates[i-1]:(cov_dates[i]-1))])
}
for(i in c(4:(length(cov_dates)))){
  monthly_r_mv[i]<-sum(daily_perform_mv[c(cov_dates[i-1]:(cov_dates[i]-1))])
}

monthly_r_1n/monthly_vol_1n*sqrt(montly_days_in_month)

data.frame()

final_data<-data.frame(dflr$Date[cov_dates],monthly_vol_1n, monthly_vol_rp, 
           monthly_vol_mv, monthly_r_1n, monthly_r_rp, monthly_r_mv, montly_days_in_month, 
           monthly_r_1n/monthly_vol_1n*sqrt(montly_days_in_month),
           monthly_r_rp/monthly_vol_rp*sqrt(montly_days_in_month),
           monthly_r_mv/monthly_vol_mv*sqrt(montly_days_in_month)
           )

setwd("~/Documents")
write.csv(final_data, file="sharpes.csv")


#Quick plots to see comparative differences in code
plot(dflr$Date[cov_dates],monthly_r_1n/monthly_vol_1n, type="l")
lines(dflr$Date[cov_dates],monthly_r_rp/monthly_vol_rp,col = "red",type = 'l')
lines(dflr$Date[cov_dates],monthly_r_mv/monthly_vol_mv,col = "blue",type = 'l')




mean(monthly_r_1n/monthly_vol_1n, na.rm = T)
mean(monthly_r_rp/monthly_vol_rp, na.rm = T)
mean(monthly_r_mv/monthly_vol_mv, na.rm = T)

sum(pmax(monthly_r_1n/monthly_vol_1n, monthly_r_rp/monthly_vol_rp, 
         monthly_r_mv/monthly_vol_mv, na.rm = T)==monthly_r_1n/monthly_vol_1n, na.rm = T)

sum(pmax(monthly_r_1n/monthly_vol_1n, monthly_r_rp/monthly_vol_rp, 
         monthly_r_mv/monthly_vol_mv, na.rm = T)==monthly_r_rp/monthly_vol_rp, na.rm = T)

sum(pmax(monthly_r_1n/monthly_vol_1n, monthly_r_rp/monthly_vol_rp, 
         monthly_r_mv/monthly_vol_mv, na.rm = T)==monthly_r_mv/monthly_vol_mv, na.rm = T)


head(monthly_r_1n/monthly_vol_1n, 20)
head(monthly_r_rp/monthly_vol_rp, 20)
head(monthly_r_mv/monthly_vol_mv, 20)


output_data<-data.frame()


