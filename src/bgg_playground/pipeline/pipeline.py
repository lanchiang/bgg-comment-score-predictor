from sagemaker.huggingface import HuggingFace
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.pipeline import Pipeline

role = 'arn:aws:iam::854139337534:role/sagemaker-general-role'
# role = sagemaker.get_execution_role()
output_path = 'file://tmp/sagemaker/models/'

local_pipeline_session = LocalPipelineSession()

# Define training hyperparameters
hyperparameters = {
    "epochs": 3,
    "batch_size": 16,
}

huggingface_estimator = HuggingFace(
    entry_point='train.py',
    source_dir='',
    instance_type='local_gpu',
    instance_count=1,
    role=role,
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
    sagemaker_session=local_pipeline_session,
    hyperparameters=hyperparameters,
    output_path=output_path
    # metric_definitions=[
    #     {'Name': 'train:loss', 'Regex': 'loss: ([0-9\\.]+)'},
    #     {'Name': 'eval:loss', 'Regex': 'eval_loss: ([0-9\\.]+)'},
    #     {'Name': 'eval:accuracy', 'Regex': 'eval_accuracy: ([0-9\\.]+)'},
    # ]
)

training_step = TrainingStep(
    name='TrainModel',
    step_args=huggingface_estimator.fit(job_name='bgg-training')
)

pipeline = Pipeline(
    name='bgg-pipeline',
    steps=[training_step],
    sagemaker_session=local_pipeline_session
)

pipeline.create(
    role_arn=role,
    description='A pipeline to train a PyTorch model for the BGG dataset'
)

# pipeline executes locally
execution = pipeline.start()

steps = execution.list_steps()

training_job_name = steps['PipelineExecutionSteps'][0]['Metadata']['TrainingJob']['Arn']

step_outputs = local_pipeline_session.sagemaker_client.describe_training_job(TrainingJobName = training_job_name)