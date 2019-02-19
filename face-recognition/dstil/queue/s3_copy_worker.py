import logging
import boto3
import botocore
import os
import sys
from dstil.worker import Worker
from dstil.util import file_utils, s3_utils


class S3CopyWorker(Worker):
    """
    A worker that takes valid file paths from an input queue and copies the files to the configured bucket on Amazon S3.
    If set, the dataset root path will be stripped from the beginning of the input paths (e.g. /mnt/theia-data/path/to/file.jpg becomes /path/to/file.jpg on S3).
    """
    def __init__(self, rabbitmq_host, rabbitmq_port, rabbitmq_username, rabbitmq_password, queue, dead_letter_queue, target_bucket, target_folder, dataset_root):
        super().__init__(rabbitmq_host, rabbitmq_port, rabbitmq_username, rabbitmq_password, queue, dead_letter_queue)

        # S3 transfer parameters.
        self.s3_client = boto3.client("s3")
        self.target_bucket = target_bucket
        self.target_folder = target_folder
        self.dataset_root = dataset_root

        # Poll bucket until found.
        s3_utils.wait_for_bucket(self.s3_client, self.target_bucket)


    def process_job(self, properties, body):
        """
        Takes a file path resolvable by this worker and copies the file to S3.
        """
        file_path = body.decode()
        s3_path = file_path.strip("/")

        # If we want a dataset-relative path on S3, strip the leading paths to the local dataset.
        if self.dataset_root is not "/":
            s3_path = file_path.replace(self.dataset_root, "", 1).strip("/")

        # Prepend target folder to S3 path.
        if self.target_folder is not "/":
            s3_path = os.path.join(self.target_folder, s3_path)

        try:
            self.logger.info("Uploading {} --> s3://{}/{}".format(file_path, self.target_bucket, s3_path))

            if not s3_utils.object_exists(self.s3_client, self.target_bucket, s3_path):
                # File doesn't exist on S3; copy it over.
                self.s3_client.upload_file(file_path, self.target_bucket, s3_path)
                self.logger.info("Upload complete.")
            else:
                # File exists; compare MD5 hash of both files.
                self.logger.info("File exists; comparing hashes...")
                response = self.s3_client.get_object(Bucket=self.target_bucket, Key=s3_path)

                local_file_hash = file_utils.md5(file_path)
                remote_file_hash = file_utils.md5_from_bytes(response["Body"].read())

                if local_file_hash == remote_file_hash:
                    self.logger.info("Files are identical; skipping.")
                else:
                    self.logger.error("Files have same name but different contents; sending to dead letter queue")
                    return False

            return True
        except FileNotFoundError as e:
            self.logger.error(e)
            return False
        except Exception as e:
            self.logger.exception(e)
            return False

    def clean_up(self):
        pass
