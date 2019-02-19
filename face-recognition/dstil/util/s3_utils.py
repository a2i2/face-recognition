import botocore
from dstil import logging

LOGGER = logging.getLogger(__name__)


def wait_for_bucket(s3_client, bucket):
    """
    Looks for the target bucket to ensure it exists and is accessible. If the bucket doesn't exist,
    the waiter will poll on a fixed period for a fixed number of attempts before throwing an exception.
    """
    LOGGER.info("Looking for bucket {}...".format(bucket))
    waiter = s3_client.get_waiter("bucket_exists")
    waiter.wait(
        Bucket=bucket,
        WaiterConfig=dict(
            Delay=5,
            MaxAttempts=20
        )
    )
    LOGGER.info("Bucket found.")


def object_exists(s3_client, bucket, key):
    """
    Returns True if the given key exists in the specified bucket.
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            # Object doesn't exist.
            return False
        else:
            # Something else went wrong.
            raise

    return True
