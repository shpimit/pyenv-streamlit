import json
import boto3

client = boto3.client('ses', region_name='us-west-2')

def lambda_handler(event, context):

    response = client.send_email(
    Destination={
        'ToAddresses': ['RecipientEmailAddress']
    },
    Message={
        'Body': {
            'Text': {
                'Charset': 'UTF-8',
                'Data': 'This is the message body in text format.',
            }
        },
        'Subject': {
            'Charset': 'UTF-8',
            'Data': 'Test email',
        },
    },
    Source='SourceEmailAddress'
    )

    print(response)

    return {
        'statusCode': 200,
        'body': json.dumps("Email Sent Successfully. MessageId is: " + response['MessageId'])
    } 