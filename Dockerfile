FROM python:2.7-alpine

RUN  apk update && apk add openjdk8 
RUN wget http://datax-opensource.oss-cn-hangzhou.aliyuncs.com/datax.tar.gz
RUN tar -zxvf datax.tar.gz && /opt && cd /opt&& mv datax /opt/ && rm datax.tar.gz
