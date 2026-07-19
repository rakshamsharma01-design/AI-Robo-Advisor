# AI-Powered Robo Advisor for Wealth Management

Course Project — Build AI with AWS

## Overview
This project implements a machine-learning-driven trading signal system with a conversational
chatbot interface, deployed entirely on serverless AWS infrastructure. It moves beyond static,
rule-based algorithmic trading by using a trained model to adapt to historical market data, and
adds a generative AI layer so recommendations are explained in plain English rather than raw output.

## Architecture
Data Sources (Yahoo Finance) → Amazon S3 (raw data) → Model training (Jupyter Notebook)
→ Amazon S3 (model parameters) → AWS Lambda (inference + explanation) → Amazon DynamoDB (storage)
→ Amazon API Gateway / Amazon Lex (user-facing interfaces) → Amazon Bedrock (natural-language explanations)

See `Architecture_Diagram.png` for the full visual.

## AWS Services Used
| Service | Purpose |
|---|---|
| Amazon S3 | Stores raw OHLCV data and trained model parameters |
| AWS Lambda | Runs model inference and serves as Lex fulfillment |
| Amazon API Gateway | Exposes Lambda as a REST endpoint |
| Amazon DynamoDB | Stores live trading signals and explanations |
| Amazon Lex | Conversational chatbot for portfolio insights |
| Amazon Bedrock | Generates natural-language trade explanations (Generative AI) |

## Data
- **Historical market data (OHLCV)**: pulled via `yfinance` for the top 50 companies by market cap
- **Features engineered**: 1-day return, 10-day and 30-day simple moving averages, 10-day
  volatility, 10-day momentum
- **Target**: whether next-day closing price is higher than the current day's close

## Models Trained & Compared
| Model | Notes |
|---|---|
| Logistic Regression | Selected for deployment — lightweight, avoids Lambda's 250MB size limit, no external ML libraries required at inference time |
| Random Forest | Trained for comparison in the notebook |

Model comparison, accuracy scores, and the cumulative return backtest are in
`Week4_Model_Training.ipynb`. The chart below compares the Logistic Regression strategy's
cumulative return against a buy-and-hold baseline:

`plots/cumulative_return_comparison.png`

## Deployment Design Decision
Rather than shipping the full scikit-learn model (with numpy/scipy dependencies) into Lambda via
a layer, this project exports just the Logistic Regression's coefficients to a small JSON file
(`model_params.json`) and performs inference with plain Python inside the Lambda function. This
avoids AWS Lambda's 250MB package size limit entirely and keeps cold-start times low, while still
using the full trained model for evaluation and comparison in the notebook.

## Generative AI Layer
Amazon Bedrock (Titan Text) is called from within the Lambda function to convert the raw
BUY/HOLD-SELL signal and confidence score into a short, plain-English explanation aimed at a
retail investor, including a brief risk disclaimer. If Bedrock is unavailable, the function falls
back to a template-based explanation so the chatbot still responds.

## Chatbot
An Amazon Lex bot (`RoboAdvisorBot`, deployed in Asia Pacific - Singapore, since Lex is not
available in all regions) collects a stock ticker from the user (e.g., "What's your
recommendation for AAPL?") via a `GetRecommendation` intent with a `Ticker` slot, and routes the
request to a dedicated fulfillment Lambda function (`RoboAdvisorLexFulfillment`). This function
reads the model parameters and writes recommendations to the same S3 bucket and DynamoDB table
used by the Week 4 API Gateway Lambda (`RoboAdvisorInference`) in Mumbai (ap-south-1), accessed
cross-region via explicit `region_name` parameters in boto3. The bot returns a natural-language
recommendation and explanation directly in the chat interface.

## Cross-Region Architecture Note
Amazon Lex is not available in every AWS region (including ap-south-1 / Mumbai, where this
project's S3 bucket, DynamoDB table, and primary Lambda function are hosted). Rather than
migrating the entire stack, a second Lambda function (`RoboAdvisorLexFulfillment`) was deployed
in ap-southeast-1 (Singapore) — a Lex-supported region — configured to read/write the Mumbai S3
and DynamoDB resources directly. This demonstrates a realistic pattern for working within AWS
regional service availability constraints without duplicating data or retraining models.

## Business Use Case
Wealth management firms currently rely on advisors and static rule-based systems to generate
trading recommendations, which do not adapt automatically as new market data arrives and require
manual re-programming to change strategy. This project demonstrates how a firm could offer
scalable, adaptive, and explainable trading insights to clients through a conversational
interface, reducing manual advisory overhead while keeping recommendations understandable and
auditable via the generated explanations stored alongside every prediction.

## Conclusions
- The Logistic Regression strategy outperformed the buy-and-hold baseline over the backtest
  period for the sampled ticker, though results vary across tickers and time windows (see
  notebook for full evaluation).
- A fully serverless architecture (S3 + Lambda + DynamoDB + API Gateway + Lex + Bedrock) is
  sufficient to deploy an end-to-end ML-driven advisory workflow without managing servers.
- Exporting lightweight model parameters instead of full model artifacts is a practical pattern
  for deploying simple models to AWS Lambda without exceeding size limits.
- AWS regional service availability (e.g., Lex not being available in every region) can be
  worked around by deploying a small fulfillment function in a supported region that reads/writes
  data resources cross-region, rather than duplicating the full stack.
- The generative AI layer (Amazon Bedrock, Nova Lite via a cross-region inference profile)
  successfully converts raw model predictions into natural-language, investor-facing explanations
  with an appropriate risk disclaimer, satisfying the project's Generative AI Features requirement.

## Repository Structure
```
├── Week4_Model_Training.ipynb        # Data pull, feature engineering, model training, backtest
├── lambda_function.py                 # Lambda: inference + DynamoDB + Bedrock explanation
├── model_params.json                  # Exported Logistic Regression coefficients (upload to S3)
├── raw_ohlcv_top50.csv                 # Raw historical data (or see notebook to regenerate)
├── plots/
│   └── cumulative_return_comparison.png
├── Architecture_Diagram.png
└── README.md
```

## How to Reproduce
1. Run `Week4_Model_Training.ipynb` top to bottom to regenerate data, train models, and export
   `model_params.json`
2. Upload `raw_ohlcv_top50.csv` and `model_params.json` to your S3 bucket
3. Deploy `lambda_function.py` to an AWS Lambda function (Python 3.11 runtime, no extra layers
   required) with environment variables `MODEL_BUCKET`, `MODEL_KEY`, `TABLE_NAME`,
   `BEDROCK_MODEL_ID`, and `RESOURCE_REGION` (only needed if deploying in a different region than
   your S3/DynamoDB resources, e.g. for Lex compatibility)
4. Create a DynamoDB table `RoboAdvisorRecommendations` with partition key `ticker`
5. Attach an API Gateway trigger to the Lambda function for REST access
6. Create an Amazon Lex bot (in a Lex-supported region) with a `Ticker` slot and set a Lambda
   function as the fulfillment code hook
