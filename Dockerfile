# Hugging Face Space (Docker SDK) / any container host. Runs the Streamlit UI in hosted-demo mode:
# visitors bring their own OpenAI key and get a private temporary folder.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH JOB_AGENT_HOSTED=1 PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .

EXPOSE 7860
# XSRF protection breaks file uploads inside the Hugging Face iframe
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
