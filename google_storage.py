from google.cloud import storage

import config

storage_client = storage.Client(config.PROJECT_NAME)
default_bucket = storage_client.get_bucket(config.DEFAULT_BUCKET_NAME)
print('Bucket {} created.'.format(default_bucket.name))


def download_blob_to_filename(source_blob_name, destination_file_name, bucket_name=config.DEFAULT_BUCKET_NAME):
    """Downloads a blob from the bucket."""

    if bucket_name == config.DEFAULT_BUCKET_NAME:
        bucket = default_bucket
    else:
        bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))
