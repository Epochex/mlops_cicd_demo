# pipeline.py
import kfp
from kfp import dsl
from kfp.dsl import PipelineParam

# 定义训练步骤
@dsl.pipeline(
    name='MNIST Training Pipeline',
    description='A simple pipeline that trains an MNIST model and uploads it to MinIO.'
)
def mnist_pipeline():
    # 训练步骤
    train_op = dsl.ContainerOp(
        name='train-model',
        image='hirschazer/mnist-train:latest',  # 使用实际 Docker Hub 用户名
        command=['python', 'train_model.py'],
        file_outputs={
            'model_path': '/model/mnist_model.pt'
        }
    )
    
    # 上传步骤
    upload_op = dsl.ContainerOp(
        name='upload-model',
        image='hirschazer/mnist-upload:latest',  # 使用实际 Docker Hub 用户名
        command=['python', 'upload_model.py'],
        arguments=[
            '--file_path', train_op.outputs['model_path']
        ]
    )
    upload_op.after(train_op)


if __name__ == '__main__':
    # 编译 Pipeline
    kfp.compiler.Compiler().compile(mnist_pipeline, 'mnist_pipeline.yaml')
