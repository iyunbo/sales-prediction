import logging as log
import os.path as path
import subprocess
import sys

from googleapiclient import discovery

import trainer.util as util

log.basicConfig(level=log.INFO, format='%(asctime)s - [%(name)s] - [%(levelname)s]: %(message)s', stream=sys.stdout)

project_name = 'sales-prediction-iyunbo'
project_id = 'projects/{}'.format(project_name)
bucket_name = 'sales-prediction-iyunbo-mlengine'
package_file = 'trainer-0.1.tar.gz'
package = path.join('dist', package_file)
job_dir = 'training_job_dir'
job_object = '/' + job_dir + '/packages/' + package_file
region = 'europe-west1'
job_id = 'xgboost_14'

log.info("building new package:", package)
subprocess.check_call(
    ['python', 'setup.py', 'sdist'],
    stderr=sys.stdout)

log.info("uploading package to cloud storage")
util.upload_blob(bucket_name, package, job_object)

log.info("submitting job:", job_id)
cloudml = discovery.build('ml', 'v1')
training_inputs = {'scaleTier': 'CUSTOM',
                   'masterType': 'n1-highcpu-64',
                   'packageUris': [
                       'gs://' + bucket_name + job_object],
                   'pythonModule': 'trainer.task',
                   'region': region,
                   'jobDir': 'gs://' + bucket_name + '/' + job_dir,
                   'runtimeVersion': '1.13',
                   'pythonVersion': '3.5'}

job_spec = {'jobId': job_id, 'trainingInput': training_inputs}

request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)

response = request.execute()
log.info("job submitted:", response)

log.info("removing package :", package)
subprocess.check_call(
    ['rm', package],
    stderr=sys.stdout)
log.info("done")
