import vertexai

from google.oauth2 import service_account
from google.auth import default


def authenticate_vertex_ai(project: str, location: str, credentials_path: str, bucket_uri: str):
    if not project:
        raise ValueError("project id is empty")

    if not location:
        raise ValueError("location is empty")

    if not bucket_uri:
        raise ValueError("bucket uri is empty")
    
    credentials = None
    # If no credentials path is provided, use the default credentials
    # Otherwise, use the service account credentials
    # When running on google cloud, the credentials are automatically set to Application Default Credentials (ADC).
    # No need to explicitly provide a credentials file.
    if not credentials_path:
        credentials, _ = default()
    else:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

    vertexai.init(
        project=project,
        location=location,
        credentials=credentials,
        staging_bucket=bucket_uri,
    )
