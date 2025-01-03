import boto3
from botocore.client import Config

def list_s3_directories(bucket_name, folder_path='', endpoint_url='https://s3.us-east-2.wasabisys.com', aws_access_key_id='', aws_secret_access_key=''):
    """
    List all directories in a specific folder of an S3 bucket.

    :param bucket_name: Name of the S3 bucket
    :param folder_path: Path to the folder in the bucket (e.g., 'parent-folder/')
    :param endpoint_url: Endpoint URL for S3-compatible service (default is Wasabi's URL)
    :param aws_access_key_id: AWS Access Key ID
    :param aws_secret_access_key: AWS Secret Access Key
    :return: A list of directories within the specified folder
    """
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=Config(signature_version='s3v4')
    )

    directories = []
    try:
        # Use a paginator to handle large buckets
        paginator = s3.get_paginator('list_objects_v2')
        operation_params = {'Bucket': bucket_name, 'Prefix': folder_path, 'Delimiter': '/'}

        for page in paginator.paginate(**operation_params):
            if 'CommonPrefixes' in page:
                for prefix in page['CommonPrefixes']:
                    directories.append(prefix['Prefix'])

    except Exception as e:
        print(f"Error listing directories: {e}")
    
    return directories

# Example usage
if __name__ == "__main__":
    bucket_name = 'ai-image-editor-webapp'
    folder_path = 'cn_train_multi_signal_CN_juggernaut_v1/'  # Add a trailing slash for folders

    # Replace with your Wasabi credentials
    aws_access_key_id = '8J824EFSZLNXXTRIDCIF'
    aws_secret_access_key = 'IwHDQrnL42iE1vo2Mvmez0YSennQXrrQXN2E4VpG'

    directories = list_s3_directories(bucket_name, folder_path, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    print("Directories:")
    for directory in directories:
        print(directory)
# cn_train_multi_signal_CN_juggernaut_v1/