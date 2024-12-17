# upload_model.py
import os
import boto3

def upload_to_minio(file_path):
    minio_client = boto3.client(
        's3',
        endpoint_url=os.getenv('MINIO_ENDPOINT_URL', 'http://45.149.207.13:9000'),  # MinIO 地址
        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minio'),                # MinIO Access Key
        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minio123'),         # MinIO Secret Key
    )
    bucket_name = 'mnist-models'
    object_name = os.path.basename(file_path)

    # 创建 Bucket（如果不存在）
    try:
        minio_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except:
        minio_client.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created.")
    
    # 上传模型
    minio_client.upload_file(file_path, bucket_name, object_name)
    print(f"Model uploaded to MinIO at: {bucket_name}/{object_name}")

if __name__ == "__main__":
    model_path = "model/mnist_model.pt"
    upload_to_minio(model_path)
