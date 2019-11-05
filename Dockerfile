FROM python:2.7-alpine

RUN apk update && apk add openjdk8 
RUN wget http://datax-opensource.oss-cn-hangzhou.aliyuncs.com/datax.tar.gz
RUN tar -zxvf datax.tar.gz && rm datax.tar.gz && mv datax /opt/ 
RUN wget https://raw.githubusercontent.com/coolyu0916/dataxdocker/master/core.json -o /opt/datax/conf/core.json
