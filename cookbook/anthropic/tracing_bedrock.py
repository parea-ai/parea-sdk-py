import os

from anthropic import AnthropicBedrock
from dotenv import load_dotenv

from parea import Parea

load_dotenv()

client = AnthropicBedrock(
    # Authenticate by either providing the keys below or use the default AWS credential providers, such as
    # using ~/.aws/credentials or the "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.
    aws_access_key="<access key>",
    aws_secret_key="<secret key>",
    # Temporary credentials can be used with aws_session_token.
    # Read more at https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp.html.
    aws_session_token="<session_token>",
    # aws_region changes the aws region to which the request is made. By default, we read AWS_REGION,
    # and if that's not present, we default to us-east-1. Note that we do not read ~/.aws/config for the region.
    aws_region="us-west-2",
)


p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_anthropic_client(client)

message = client.messages.create(model="anthropic.claude-3-5-sonnet-20240620-v1:0", max_tokens=256, messages=[{"role": "user", "content": "Hello, world"}])
print(message.content)
