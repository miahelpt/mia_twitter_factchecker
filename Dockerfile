#
# NOTE: THIS DOCKERFILE IS GENERATED VIA "apply-templates.sh"
#
# PLEASE DO NOT EDIT IT DIRECTLY.
#

#FROM buildpack-deps:bullseye

FROM public.ecr.aws/lambda/python:3.8 as build
RUN yum install gcc-c++ -y
RUN pip3 install --upgrade pip

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./* /miaapp/
COPY ./classifiers/ /miaapp/classifiers/
COPY ./factcheck/ /miaapp/factcheck/
COPY ./helpers/ /miaapp/helpers/
COPY ./loaders/ /miaapp/loaders/
COPY ./lookup/ /miaapp/lookup/
COPY ./streamers/ /miaapp/streamers/
COPY ./models /miaapp/models/

RUN pip install -r /miaapp/requirements.txt  --target /miaapp/
#RUN ${LAMBDA_TASK_ROOT}/models/download.sh


FROM public.ecr.aws/lambda/python:3.8 as main
COPY --from=build /miaapp/ ${LAMBDA_TASK_ROOT}/

RUN chmod 777 ${LAMBDA_TASK_ROOT} -R

# Copy function code
#COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_app.handler" ] 