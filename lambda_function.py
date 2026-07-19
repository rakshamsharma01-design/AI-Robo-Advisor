"""
Lambda function: Robo Advisor inference + Lex/API fulfillment + Bedrock explanation
Course Project - Build AI with AWS | Week 4 / Week 5

Deploy this as your Lambda function code (Python 3.11 runtime, NO extra layers needed).
It:
  1. Loads model_params.json from S3 (Logistic Regression coefficients - no scikit-learn
     or joblib needed inside Lambda, keeping the package small and dependency-free).
  2. Computes a recommendation for a requested ticker using plain Python (manual sigmoid).
  3. Writes the recommendation to DynamoDB.
  4. Calls Amazon Bedrock to turn the raw prediction into a natural-language explanation
     (this is the Generative AI layer for Week 5).
  5. Returns a response usable by both API Gateway (Week 4) and Lex (Week 5).

Environment variables to set in the Lambda console:
  MODEL_BUCKET   = your S3 bucket name, e.g. robo-advisor-yourname-2026
  MODEL_KEY      = models/model_params.json
  TABLE_NAME     = RoboAdvisorRecommendations
  BEDROCK_MODEL_ID = amazon.titan-text-express-v1   (or another model you've enabled access to)
"""

import json
import os
import math
import boto3

# The Lambda function itself can run in any region (e.g. Singapore, for Lex compatibility),
# while still reading/writing S3 and DynamoDB resources that live in Mumbai (ap-south-1).
# Bedrock is called in the Lambda's own region.
RESOURCE_REGION = os.environ.get("RESOURCE_REGION", "ap-south-1")

s3 = boto3.client("s3", region_name=RESOURCE_REGION)
dynamodb = boto3.resource("dynamodb", region_name=RESOURCE_REGION)
bedrock = boto3.client("bedrock-runtime")  # uses the Lambda's own region

MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
MODEL_KEY = os.environ.get("MODEL_KEY", "models/model_params.json")
TABLE_NAME = os.environ.get("TABLE_NAME", "RoboAdvisorRecommendations")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "apac.amazon.nova-lite-v1:0")

_params = None  # cached across warm Lambda invocations


def load_params():
    global _params
    if _params is None:
        obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
        _params = json.loads(obj["Body"].read())
    return _params


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict(features):
    """
    features: list of 5 floats, in the SAME order as model_params['feature_order']:
    [return_1d, sma_10, sma_30, volatility_10, momentum_10]
    """
    params = load_params()
    coefs = params["coefficients"]
    intercept = params["intercept"]

    z = intercept + sum(c * f for c, f in zip(coefs, features))
    prob_up = sigmoid(z)
    return 1 if prob_up >= 0.5 else 0, prob_up


def generate_explanation(ticker, signal, prob):
    """Call Amazon Bedrock (Converse API) to turn the raw prediction into a plain-English
    explanation. Falls back to a template string if Bedrock is unavailable (e.g. model
    access not yet approved) so the demo still works end-to-end."""
    prompt = (
        f"You are a financial assistant. In 2-3 short sentences, explain to a retail "
        f"investor why a trading model recommends '{signal}' for {ticker}, given a "
        f"{prob:.0%} model confidence that the price will rise tomorrow. "
        f"Be concise, plain-English, and include a brief risk disclaimer."
    )
    try:
        response = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 200, "temperature": 0.5, "topP": 0.9},
        )
        return response["output"]["message"]["content"][0]["text"].strip()
    except Exception as e:
        print(f"Bedrock call failed, using fallback explanation: {e}")
        return (
            f"The model recommends {signal} for {ticker} with {prob:.0%} confidence "
            f"based on recent price momentum and volatility. This is not financial "
            f"advice - please do your own research before trading."
        )


def get_recommendation(ticker, features):
    pred, prob = predict(features)
    signal = "BUY" if pred == 1 else "HOLD/SELL"
    explanation = generate_explanation(ticker, signal, prob)

    table = dynamodb.Table(TABLE_NAME)
    table.put_item(Item={
        "ticker": ticker,
        "signal": signal,
        "probability_up": str(round(prob, 4)),
        "explanation": explanation,
    })

    return signal, prob, explanation


def lambda_handler(event, context):
    # --- Case 1: called via API Gateway (Week 4 test) ---
    if "body" in event:
        body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        ticker = body.get("ticker", "UNKNOWN")
        features = body.get("features", [0, 0, 0, 0, 0])
        signal, prob, explanation = get_recommendation(ticker, features)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "ticker": ticker,
                "recommendation": signal,
                "probability_up": round(prob, 4),
                "explanation": explanation,
            }),
        }

    # --- Case 2: called via Amazon Lex (Week 5) ---
    if "sessionState" in event:
        slots = event["sessionState"]["intent"]["slots"]
        ticker = slots.get("Ticker", {}).get("value", {}).get("interpretedValue", "AAPL")
        # Placeholder features for demo purposes - replace with a live feature lookup
        signal, prob, explanation = get_recommendation(ticker, [0.001, 150, 148, 0.02, 2.5])

        message = f"{signal} for {ticker} ({prob:.0%} confidence). {explanation}"

        return {
            "sessionState": {
                "dialogAction": {"type": "Close"},
                "intent": {
                    "name": event["sessionState"]["intent"]["name"],
                    "state": "Fulfilled",
                },
            },
            "messages": [{"contentType": "PlainText", "content": message}],
        }

    return {"statusCode": 400, "body": "Unrecognized event format"}
