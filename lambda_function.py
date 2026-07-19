"""
Lambda function: Robo Advisor inference + Lex/API fulfillment
Course Project — Build AI with AWS | Week 4 / Week 5

Deploy this as your Lambda function code (Python 3.11 runtime, NO extra layers needed).
It:
  1. Loads model_params.json from S3 (a tiny file - coefficients from a Logistic Regression
     model trained in the notebook). This avoids needing scikit-learn/joblib inside Lambda,
     which can blow past AWS's 250MB layer size limit.
  2. Computes a recommendation for a requested ticker using plain Python (manual sigmoid).
  3. Writes the recommendation to DynamoDB.
  4. Returns a response usable by both API Gateway (Week 4) and Lex (Week 5).

Environment variables to set in the Lambda console:
  MODEL_BUCKET   = your S3 bucket name, e.g. robo-advisor-yourname-2026
  MODEL_KEY      = models/model_params.json
  TABLE_NAME     = RoboAdvisorRecommendations
"""

import json
import os
import math
import boto3

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
MODEL_KEY = os.environ.get("MODEL_KEY", "models/model_params.json")
TABLE_NAME = os.environ.get("TABLE_NAME", "RoboAdvisorRecommendations")

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


def get_recommendation(ticker, features):
    pred, prob = predict(features)
    signal = "BUY" if pred == 1 else "HOLD/SELL"

    table = dynamodb.Table(TABLE_NAME)
    table.put_item(Item={
        "ticker": ticker,
        "signal": signal,
        "probability_up": str(round(prob, 4)),
    })

    return signal, prob


def lambda_handler(event, context):
    # --- Case 1: called via API Gateway (Week 4 test) ---
    if "body" in event:
        body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        ticker = body.get("ticker", "UNKNOWN")
        features = body.get("features", [0, 0, 0, 0, 0])
        signal, prob = get_recommendation(ticker, features)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "ticker": ticker,
                "recommendation": signal,
                "probability_up": round(prob, 4),
            }),
        }

    # --- Case 2: called via Amazon Lex (Week 5) ---
    if "sessionState" in event:
        slots = event["sessionState"]["intent"]["slots"]
        ticker = slots.get("Ticker", {}).get("value", {}).get("interpretedValue", "AAPL")
        # Placeholder features for demo purposes - replace with a live feature lookup
        signal, prob = get_recommendation(ticker, [0.001, 150, 148, 0.02, 2.5])

        message = f"Based on the latest model, my recommendation for {ticker} is {signal} (confidence {prob:.0%})."

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
