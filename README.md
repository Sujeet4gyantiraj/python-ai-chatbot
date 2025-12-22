# python-ai-chatbot
This repository contains a production-ready AI service for building SaaS-based chatbots, implemented using Python and FastAPI. It is designed as a dedicated AI microservice, responsible only for AI logic and intelligence, while remaining fully decoupled from SaaS business logic.

# pm2 start python3 --name python-ai-service -- -m uvicorn main:app --port 9002